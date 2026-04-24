#!/usr/bin/env python3

import carla
from carla import ColorConverter as cc
import argparse
import random
import sys
import time
import datetime
import weakref

try:
    import pygame
    from pygame.locals import *
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if pygame.font.match_font('courier') else pygame.font.get_default_font()
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0] if fonts else font_name
        self._font_mono = pygame.font.SysFont(mono, 12 if width < 1280 else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        v = world.player.get_velocity()
        c = world.player.get_control()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        self._info_text = [
            'Server:  % 16.1f FPS' % self.server_fps,
            'Client:  % 16.1f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % world.player.type_id,
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % speed,
            '',
            'Throttle:% 15.2f' % c.throttle,
            'Steer:   % 15.2f' % c.steer,
            'Brake:   % 15.2f' % c.brake,
            'Reverse: % 15s' % c.reverse,
            'Handbrake:% 14s' % c.hand_brake,
            'Gear:    % 15d' % c.gear,
        ]

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            for n, line in enumerate(self._info_text):
                rect = pygame.Rect(10, v_offset, 200, 15)
                surface = self._font_mono.render(line, True, (255, 255, 255))
                display.blit(surface, rect)
                v_offset += 16
        self._notifications.render(display)

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)

    def render(self, display):
        if self.seconds_left > 0.0:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, client, hud, args):
        self.client = client
        self.world = client.get_world()
        self.actor_role_name = args.rolename
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.camera_manager = None
        self._actor_filter = args.filter
        self.args = args
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        # Target 60Hz physics to match RTX 4090 / 60 FPS target
        settings.fixed_delta_seconds = 1.0 / 60.0
        self.world.apply_settings(settings)

        self.restart()
        self.spawn_npcs()
        self.spawn_pedestrians()
        self.world.on_tick(self.hud.on_world_tick)

    def spawn_pedestrians(self):
        """Spawn walkers on sidewalks throughout the map."""
        blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        spawn_points = []
        for _ in range(self.args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprints)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if random.random() > 0.5:
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                walker_speed.append(0.0)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2

        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id

        # 4. we put together the walkers and controllers id to get the objects from them
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        # 5. initialize each controller and set target to walk to
        # REDUCED CROSS FACTOR: Pedestrians will cross much less frequently (0.01)
        self.world.set_pedestrians_cross_factor(0.01)
        
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
            
            # AI BEHAVIOR: Force them to wait at red lights if possible
            # Note: 0.9.16 AI controller has basic light awareness when cross_factor is low
        
        print(f"[World] Spawned {len(self.walkers_list)} pedestrians.")

    def spawn_npcs(self):
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        # Hybrid mode ON with standard radius for reliability
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(50.0) 
        
        # CARLA DEFAULT SMOOTH TRAFFIC
        traffic_manager.set_global_distance_to_leading_vehicle(5.0) # Safe distance
        traffic_manager.global_percentage_speed_difference(0.0)    # Speed limit adherence
        
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        # STRICT SPAWN EXCLUSION: No cars within 120m of the workzone anchor
        WZ_ANCHOR = carla.Location(x=40.0, y=137.5, z=0.5) 
        all_spawn_points = self.map.get_spawn_points()
        spawn_points = [sp for sp in all_spawn_points if sp.location.distance(WZ_ANCHOR) > 120.0]
        random.shuffle(spawn_points)

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= self.args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            batch.append(carla.command.SpawnActor(blueprint, transform)
                .then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port())))

        for response in self.client.apply_batch_sync(batch, False):
            if not response.error:
                self.vehicles_list.append(response.actor_id)

    def _apply_traffic_guard(self):
        """Standard behavior enabled (No forced lane changes)."""
        pass

    def clear_npcs(self):
        """Remove all spawned NPC vehicles."""
        if self.vehicles_list:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
            self.vehicles_list.clear()
            print("[World] All NPC vehicles removed. You have the city to yourself.")

    def restart(self):
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        
        # Custom spawn point for Town10 workzone area
        spawn_point = carla.Transform(carla.Location(x=104.0, y=180.0, z=0.5), carla.Rotation(yaw=-90))

        if self.player is not None:
            self.destroy()

        while self.player is None:
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            if self.player is None:
                spawn_point = random.choice(self.map.get_spawn_points())

        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.set_sensor(0)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        # Stop and destroy walkers
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()
        
        if self.all_id:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

        if self.camera_manager is not None and self.camera_manager.sensor is not None:
            self.camera_manager.sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        if self.vehicles_list:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

# ==============================================================================
# -- Logitech Control ----------------------------------------------------------
# ==============================================================================

class LogitechControl(object):
    def __init__(self, world):
        self._world = world
        self._control = carla.VehicleControl()
        pygame.joystick.init()
        self._joystick = None
        self._ffb_device = None
        
        count = pygame.joystick.get_count()
        print(f"[Debug] Pygame sees {count} joysticks.")
        
        for i in range(count):
            j = pygame.joystick.Joystick(i)
            j.init()
            name = j.get_name()
            print(f"[Debug] Checking Joystick {i}: {name}")
            if "G920" in name or "G29" in name or "Driving Force" in name:
                self._joystick = j
                print(f"[Debug] Selected G920/G29: {name}")
                break
        
        if self._joystick is None and count > 0:
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            print(f"[Debug] Fallback to Joystick 0: {self._joystick.get_name()}")

        self._setup_ffb()

    def _setup_ffb(self):
        """Setup Force Feedback for Linux using evdev."""
        try:
            import evdev
            from evdev import ecodes
            
            # Find the event device path for G920
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            for device in devices:
                if "G920" in device.name or "G29" in device.name or "Driving Force" in device.name:
                    self._ffb_device = device
                    print(f"[FFB] Found device for Force Feedback: {device.path}")
                    
                    # Set Autocenter force (Basic FFB)
                    # 0xFFFF is maximum strength (65535)
                    # We'll start with 25% (16383) and modulate later
                    device.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, 16383)
                    break
        except Exception as e:
            print(f"[FFB] Could not setup Force Feedback: {e}")
            print("[FFB] Hint: Try running 'sudo chmod 666 /dev/input/event*' if it is a permission error.")

    def _update_ffb(self):
        """Update Force Feedback strength based on vehicle speed."""
        if self._ffb_device is None:
            return
            
        try:
            from evdev import ecodes
            v = self._world.player.get_velocity()
            speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
            
            # Increased strength for more feedback: 15% at rest, 55% at speed
            strength = int(65535 * (0.15 + min(speed / 100.0, 0.40)))
            self._ffb_device.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, strength)
        except:
            pass

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            
            # Debug: Print ALL joystick events
            if event.type == pygame.JOYBUTTONDOWN:
                print(f"[Debug] Button {event.button} pressed")
                if event.button == 9: # Start/Options
                    world.restart()
                if event.button == 1: # B / Red Button
                    world.clear_npcs()
                if event.button == 0: # A / X Button
                    self._control.reverse = not self._control.reverse
                if event.button == 8: # View / Share
                    world.camera_manager.toggle_camera()
            
            elif event.type == pygame.JOYAXISMOTION:
                pass # Silent axis motion to avoid spam

            elif event.type == pygame.KEYUP:
                if event.key == K_ESCAPE:
                    return True

        if self._joystick:
            self._parse_joystick()
            self._update_ffb()
        self._world.player.apply_control(self._control)

    def _parse_joystick(self):
        # Specific G920 Linux Mapping (4 axes: 0=Steer, 1=Throttle, 2=Brake, 3=Clutch)
        if "G920" in self._joystick.get_name():
            # Inverted mapping: 1.0 (rest) to -1.0 (pressed)
            self._control.throttle = (1.0 - self._joystick.get_axis(1)) / 2.0
            
            # BRAKE BOOST: The G920 brake is very stiff on Linux. 
            # We multiply by 4.0 to reach 100% braking with realistic foot pressure.
            raw_brake = (1.0 - self._joystick.get_axis(2)) / 2.0
            self._control.brake = min(1.0, raw_brake * 4.0)
            
            self._control.steer = self._joystick.get_axis(0)
            
            # Deadzone to prevent ghost inputs
            if self._control.throttle < 0.05: self._control.throttle = 0.0
            if self._control.brake < 0.05: self._control.brake = 0.0
            
            # Debug: ONLY print when braking to verify boost
            if self._control.brake > 0.1:
                print(f"[Input] BRAKE APPLIED: {self._control.brake:.2f} (Raw: {raw_brake:.2f})")
            
        else:
            # Generic/G29 Mapping
            self._control.throttle = (self._joystick.get_axis(5) + 1.0) / 2.0 if self._joystick.get_numaxes() > 5 else 0.0
            self._control.brake = (self._joystick.get_axis(2) + 1.0) / 2.0 if self._joystick.get_numaxes() > 2 else 0.0
            self._control.steer = self._joystick.get_axis(0) * 0.7

        # Force handbrake release when throttle is applied
        if self._control.throttle > 0.1:
            self._control.hand_brake = False
        # if self._control.throttle > 0.01 or self._control.brake > 0.01:
        #      gear = self._world.player.get_control().gear
        #      print(f"[Vehicle] Throttle: {self._control.throttle:.2f} | Brake: {self._control.brake:.2f} | Gear: {gear}")

# ==============================================================================
# -- Camera Manager ------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.index = 0
        
        # Dashcam position updated for realism
        self.transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=0.45, y=-0.4, z=1.18), carla.Rotation(pitch=-8)),
            # PRECISE DASHCAM: Top of windshield, looking down at the hood
            carla.Transform(carla.Location(x=0.8, y=0.0, z=1.55), carla.Rotation(pitch=-15))
        ]
        
        world = self._parent.get_world()
        self.blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        self.blueprint.set_attribute('image_size_x', str(hud.dim[0]))
        self.blueprint.set_attribute('image_size_y', str(hud.dim[1]))
        self.blueprint.set_attribute('fov', '90')

    def set_sensor(self, index):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
        self.index = index
        self.sensor = self._parent.get_world().spawn_actor(
            self.blueprint,
            self.transforms[self.index],
            attach_to=self._parent)
        
        # PERSISTENT SENSOR: Ensure the sensor object is stored to prevent garbage collection
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

    def toggle_camera(self):
        self.set_sensor((self.index + 1) % len(self.transforms))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self: return
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

# ==============================================================================
# -- Game Loop -----------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    world = None
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0) 
        
        print("[Logitech] Connecting to existing world...")
        sim_world = client.get_world()
        
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        display = pygame.display.set_mode((args.width, args.height), flags)
        
        hud = HUD(args.width, args.height)
        # Pass the existing world directly
        world = World(client, hud, args)
        controller = LogitechControl(world)
        clock = pygame.time.Clock()
        
        print("[Logitech] Connected! Ready for steering.")
        
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock):
                return
            
            # Apply traffic monitoring logic to keep NPCs away from the workzone
            world._apply_traffic_guard()
            
            world.world.tick()
            world.hud.tick(world, clock)
            world.render(display)
            pygame.display.flip()
    except Exception as e:
        print(f"[Logitech] Error during connection: {e}")
    finally:
        if world is not None:
            world.destroy()
        pygame.quit()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='CARLA Logitech Manual Control')
    argparser.add_argument('--host', default='127.0.0.1')
    argparser.add_argument('--port', default=2000, type=int)
    argparser.add_argument('--res', default='1280x720')
    argparser.add_argument('--filter', default='vehicle.tesla.model3')
    argparser.add_argument('--rolename', default='hero')
    argparser.add_argument('-n', '--number-of-vehicles', default=30, type=int)
    argparser.add_argument('-w', '--number-of-walkers', default=10, type=int)
    argparser.add_argument('--map', default='Town10HD_Opt')
    argparser.add_argument('--fullscreen', action='store_true', help='Open in fullscreen mode')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    game_loop(args)
