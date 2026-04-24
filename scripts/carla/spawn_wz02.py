#!/usr/bin/env python3
"""
spawn_wz02.py — Realistic Urban Workzone (WZ-02)
MUTCD Part 6, urban 25 mph compliant layout.

ESV 2026 Demo (Mi3 Lab)  |  Town10HD_Opt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import random

import carla

LOG = logging.getLogger('wz02_v5')

# ── Map & Anchor ──────────────────────────────────────────────────────────────
MAP_NAME   = 'Town10HD_Opt'
# NEW ANCHOR: Road 1, a long straight avenue (>100m).
# Anchor is on Lane -1 (inner driving lane, y=137.5).
# We are closing Lane -2 (outer driving lane / right lane, y=141.0).
ANCHOR     = carla.Location(x=40.0, y=137.5, z=0.5)
LANE_WIDTH = 3.5  # meters

# ── MUTCD Urban 25 mph Layout Constants ───────────────────────────────────────
SIGN_DISTS        = (60.0, 8.0)      # First sign moved to 25m (after police car)
TAPER_CONES       = 12                # Doubled for density
TAPER_SPACING     = 3.5               # Halved spacing (12 * 3.5 = 42m)
BUFFER            = 12.0              # Clear space before work
BARRIERS          = 5                 # Work area span
BARRIER_GAP       = 4.0               # 5 * 4 = 20m work area
WORKER_OFFSETS    = (5.0, 13.0)
TERM_CONES        = 6                 # Doubled for density
TERM_SPACING      = 2.5               # Halved spacing (6 * 2.5 = 15m)

# Lateral offsets (Positive = Driver's Right)
OFF_SIGN_URBAN    = +(LANE_WIDTH * 2.0)   # +7.0m (Move signs to the edge/sidewalk)
OFF_BARRIER       = +(LANE_WIDTH / 2.0)   # +1.75m (Lane Divider Line)
OFF_WORKER        = +(LANE_WIDTH + 0.5)   # +4.0m (Middle of closed lane)

# ── Blueprints ──────────────────────────────────────────────────────────────
BP_CONES = ['static.prop.constructioncone', 'static.prop.trafficcone01']
BP_BARRIER = 'static.prop.streetbarrier'
BP_WARNING = 'static.prop.warningconstruction'
BP_SPEED_30 = 'static.prop.trafficwarning'  # Valid fallback for general warning
BP_TRUCK = 'vehicle.mercedes.sprinter'
BP_POLICE_CAR = 'vehicle.dodge.charger_police'

# Blueprints for different roles
_POLICE_IDS = ['walker.pedestrian.0030', 'walker.pedestrian.0032']
_WORKER_IDS = ['walker.pedestrian.0052']

# ── Global Cleanup ────────────────────────────────────────────────────────────

def purge_stale_actors(world: carla.World) -> int:
    """Destroy ALL leftover vehicles, walkers, and construction props."""
    count = 0
    # Clean static props, vehicles and walkers
    for a in world.get_actors():
        is_prop = a.type_id.startswith('static.prop.') and any(kw in a.type_id for kw in ('cone', 'barrier', 'warningconstruction'))
        is_vehicle = a.type_id.startswith('vehicle.')
        is_walker = a.type_id.startswith('walker.')

        if is_prop or is_vehicle or is_walker:
            try:
                a.destroy()
                count += 1
            except Exception:
                pass

    # NEW: Hide static environment cars that block the signage
    objs = world.get_environment_objects(carla.CityObjectLabel.Car)
    # Also hide motorcycles if any
    try:
        objs += world.get_environment_objects(carla.CityObjectLabel.Motorcycle)
    except:
        pass

    to_hide = [o.id for o in objs if o.transform.location.distance(ANCHOR) < 60.0]
    if to_hide:
        world.enable_environment_objects(to_hide, False)
        LOG.info("Hid %d static parked vehicles (cars/motos) in the entire workzone area.", len(to_hide))

    if count > 0:
        LOG.info("Purged %d stale actors (including vehicles/walkers) from map.", count)
    return count
# ── Workzone Spawner ─────────────────────────────────────────────────────────

class WorkzoneSpawner:
    def __init__(self, world: carla.World) -> None:
        self.world = world
        self._map = world.get_map()
        self._lib = world.get_blueprint_library()
        self._actors: list[carla.Actor] = []

        self._anchor = self._map.get_waypoint(
            ANCHOR,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if self._anchor is None:
            raise RuntimeError(f"No drivable waypoint near anchor {ANCHOR}.")

    def _ahead(self, wp: carla.Waypoint, dist: float) -> carla.Waypoint:
        if dist <= 0.0: return wp
        nxt = wp.next(dist)
        return nxt[0] if nxt else wp

    def _behind(self, wp: carla.Waypoint, dist: float) -> carla.Waypoint:
        if dist <= 0.0: return wp
        prv = wp.previous(dist)
        return prv[0] if prv else wp

    def _right_offset(self, wp: carla.Waypoint, lateral: float, z_lift: float = 0.1) -> carla.Location:
        r = wp.transform.get_right_vector()
        base = wp.transform.location
        return carla.Location(
            x=base.x + r.x * lateral,
            y=base.y + r.y * lateral,
            z=base.z + z_lift,
        )

    def _spawn(self, bp_id: str, loc: carla.Location, rot: carla.Rotation, 
               pos_jitter: float = 0.0) -> carla.Actor | None:
        bp = self._lib.find(bp_id)
        if not bp: return None
        if pos_jitter > 0.0:
            loc.x += random.uniform(-pos_jitter, pos_jitter)
            loc.y += random.uniform(-pos_jitter, pos_jitter)
        
        actor = self.world.try_spawn_actor(bp, carla.Transform(loc, rot))
        if actor:
            self._actors.append(actor)
        return actor

    def build(self) -> None:
        LOG.info("╔═ WZ-02 Urban Lane Closure (MUTCD Compliant) ═══════════════╗")

        # 1. Advance Warning Area (Upstream)
        # Park Police Car BEFORE the first sign (at 30m upstream)
        wp_police_adv = self._behind(self._anchor, 30.0)
        # Rotate 180 to face oncoming traffic
        rot_police_adv = carla.Rotation(yaw=wp_police_adv.transform.rotation.yaw + 180.0)
        # MOVED to the right
        police_car = self._spawn(BP_POLICE_CAR, self._right_offset(wp_police_adv, OFF_WORKER + 2.0), rot_police_adv)
        if police_car and isinstance(police_car, carla.Vehicle):
            police_car.set_light_state(carla.VehicleLightState(
                carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2 | carla.VehicleLightState.Position
            ))
        
        # Place the guarding officer near the police car (moved right)
        self._spawn(random.choice(_POLICE_IDS), self._right_offset(wp_police_adv, OFF_SIGN_URBAN), rot_police_adv)
        
        # NEW: Officer standing in FRONT of the police car's headlights (33m upstream)
        wp_officer_front = self._behind(self._anchor, 33.0)
        rot_officer_front = carla.Rotation(yaw=wp_officer_front.transform.rotation.yaw + 180.0)
        # Alinhado com o carro (OFF_WORKER + 2.0)
        self._spawn(random.choice(_POLICE_IDS), self._right_offset(wp_officer_front, OFF_WORKER + 2.0), rot_officer_front)
        
        LOG.info("  advance police ✓  parked with 2 officers guarding")

        # 1. Advance Warning Area (Upstream) - SPAN INDIVIDUALLY
        # Sign 1: 50m upstream (20m BEFORE the police car)
        wp_50 = self._behind(self._anchor, 50.0)
        loc_50 = self._right_offset(wp_50, OFF_SIGN_URBAN)
        rot_50 = carla.Rotation(yaw=wp_50.transform.rotation.yaw + 90.0)
        self._spawn(BP_WARNING, loc_50, rot_50)
        LOG.info("  placed first sign at 50m (advance warning)")

        # Sign 2: 15m upstream, on the RIGHT (Visible after the police car)
        wp_15 = self._behind(self._anchor, 15.0)
        loc_15 = self._right_offset(wp_15, 5.0) # MOVED A LITTLE MORE TO THE LEFT
        rot_15 = carla.Rotation(yaw=wp_15.transform.rotation.yaw + 90.0)
        self._spawn(BP_WARNING, loc_15, rot_15)
        LOG.info("  placed second sign at 15m on the right (slightly left of the first)")

        # Message Board (Billboard): 2m upstream, on the RIGHT
        wp_msg = self._behind(self._anchor, 2.0)
        loc_msg = self._right_offset(wp_msg, OFF_SIGN_URBAN)
        rot_msg = carla.Rotation(yaw=wp_msg.transform.rotation.yaw - 90.0)
        self._spawn(BP_SPEED_30, loc_msg, rot_msg)

        LOG.info("  all signs placed individually for maximum visibility")

        # 2. Taper Area (Closes the right lane)
        wp = self._anchor
        for i in range(TAPER_CONES):
            if i > 0:
                wp = self._ahead(wp, TAPER_SPACING)
            t = i / max(TAPER_CONES - 1, 1)
            lateral = (LANE_WIDTH / 2.0) + LANE_WIDTH * (1.0 - t)
            self._spawn(random.choice(BP_CONES), self._right_offset(wp, lateral), wp.transform.rotation, pos_jitter=0.05)
        taper_end = TAPER_CONES * TAPER_SPACING
        LOG.info("  taper ✓  correctly pushes traffic from right to left (%.1fm)", taper_end)

        # 3. Buffer Zone (with tangent cones and police car)
        work_start = taper_end + BUFFER
        
        # Fill the buffer space with cones up to the barrier
        wp_buffer = self._ahead(self._anchor, taper_end)
        buffer_cones = int(BUFFER / 2.0)
        for i in range(1, buffer_cones + 1):
            wp_buffer = self._ahead(wp_buffer, 2.0)
            self._spawn(random.choice(BP_CONES), self._right_offset(wp_buffer, OFF_BARRIER), wp_buffer.transform.rotation, pos_jitter=0.05)
        
        LOG.info("  buffer ✓  clear zone of %.1fm with continuous cones", BUFFER)

        # 4. Work Area (Barriers parallel to the lane line)
        for i in range(BARRIERS):
            dist = work_start + i * BARRIER_GAP
            wp = self._ahead(self._anchor, dist)
            self._spawn(BP_BARRIER, self._right_offset(wp, OFF_BARRIER), wp.transform.rotation)
        work_end = work_start + BARRIERS * BARRIER_GAP
        LOG.info("  work area ✓  protected by %d barriers", BARRIERS)

        # Work Vehicle & Workers inside the closed lane
        mid_work = work_start + (BARRIERS * BARRIER_GAP) * 0.4
        wp_mid = self._ahead(self._anchor, mid_work)
        truck_loc = self._right_offset(wp_mid, OFF_WORKER)
        truck = self._spawn(BP_TRUCK, truck_loc, wp_mid.transform.rotation)
        if truck and isinstance(truck, carla.Vehicle):
            truck.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.Special1))

        # NEW: Officer standing in the MIDDLE of the workzone (next to barriers)
        wp_officer_mid = self._ahead(self._anchor, work_start + (BARRIERS * BARRIER_GAP) / 2.0)
        rot_officer_mid = carla.Rotation(yaw=wp_officer_mid.transform.rotation.yaw + 180.0)
        self._spawn(random.choice(_POLICE_IDS), self._right_offset(wp_officer_mid, OFF_BARRIER + 0.5), rot_officer_mid)

        # We'll place high-visibility workers specifically in the work zone near the truck
        for offset in WORKER_OFFSETS:
            wp = self._ahead(self._anchor, work_start + offset)
            rot = carla.Rotation(yaw=wp.transform.rotation.yaw + 180.0)
            self._spawn(random.choice(_WORKER_IDS), self._right_offset(wp, OFF_WORKER), rot)

        # NEW: Two workers near the Bus Station area
        # Worker 1: 6m ahead
        wp_bus1 = self._ahead(self._anchor, 6.0)
        rot_bus1 = carla.Rotation(yaw=wp_bus1.transform.rotation.yaw + 180.0) 
        self._spawn(random.choice(_WORKER_IDS), self._right_offset(wp_bus1, OFF_SIGN_URBAN), rot_bus1)
        
        # Worker 2: 16m ahead (10m away from the first)
        wp_bus2 = self._ahead(self._anchor, 16.0)
        rot_bus2 = carla.Rotation(yaw=wp_bus2.transform.rotation.yaw + 180.0)
        self._spawn(random.choice(_WORKER_IDS), self._right_offset(wp_bus2, OFF_SIGN_URBAN), rot_bus2)
        
        LOG.info("  bus station ✓  2 additional workers placed (separated by 10m)")

        # 5. Termination Taper (Re-opens the right lane)
        wp = self._ahead(self._anchor, work_end)
        for i in range(TERM_CONES):
            if i > 0:
                wp = self._ahead(wp, TERM_SPACING)
            t = i / max(TERM_CONES - 1, 1)
            lateral = (LANE_WIDTH / 2.0) + LANE_WIDTH * t
            self._spawn(random.choice(BP_CONES), self._right_offset(wp, lateral), wp.transform.rotation, pos_jitter=0.05)
        
        LOG.info("╚═ WZ-02 complete ══════════════════════════════════════════════╝")

    def position_spectator(self) -> None:
        wp = self._behind(self._anchor, 30.0)
        loc = self._right_offset(wp, -5.0, z_lift=15.0)
        yaw = self._anchor.transform.rotation.yaw
        self.world.get_spectator().set_transform(carla.Transform(loc, carla.Rotation(pitch=-25.0, yaw=yaw)))

    def cleanup(self) -> None:
        count = 0
        for actor in self._actors:
            if actor.is_alive:
                actor.destroy()
                count += 1
        self._actors.clear()
        LOG.info("Local cleanup: destroyed %d actors", count)


def setup_environment(world: carla.World, night: bool = False) -> None:
    """Configure weather and lighting."""
    if night:
        # Realistic Night: Deep darkness, wet ground for emergency light reflections
        weather = carla.WeatherParameters(
            cloudiness=15.0,
            precipitation=0.0,
            sun_altitude_angle=-80.0,
            sun_azimuth_angle=0.0,
            precipitation_deposits=60.0,
            wetness=60.0,
            fog_density=2.0,
            mie_scattering_scale=0.03,
            rayleigh_scattering_scale=0.0331,
            scattering_intensity=1.0
        )
        LOG.info("Environment: Night mode enabled (with wetness for reflections)")
    else:
        # Clear Day
        weather = carla.WeatherParameters.ClearNoon
        LOG.info("Environment: Clear Day mode")
    
    world.set_weather(weather)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--night', action='store_true', help='Enable night mode with wetness reflections')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    
    if not world.get_map().name.endswith(MAP_NAME):
        world = client.load_world(MAP_NAME)
        time.sleep(2.0)

    # Set environment (Night/Day)
    setup_environment(world, args.night)

    # Global cleanup before doing anything
    purge_stale_actors(world)

    spawner = WorkzoneSpawner(world)
    
    try:
        spawner.build()
        spawner.position_spectator()
        LOG.info("Workzone deployed. Press Ctrl+C to exit and cleanup.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        spawner.cleanup()
        return 0

if __name__ == '__main__':
    sys.exit(main())