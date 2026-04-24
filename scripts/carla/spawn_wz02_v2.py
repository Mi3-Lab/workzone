#!/usr/bin/env python3
"""
spawn_wz02_v2.py — Realistic Urban Workzone (WZ-02)
Improved version with randomization, debris, and traffic awareness.

ESV 2026 Demo (Mi3 Lab)  |  Town10HD_Opt  |  MUTCD Part 6, urban 25 mph
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import random

import carla

LOG = logging.getLogger('wz02_v2')

# ── Map & Anchor ──────────────────────────────────────────────────────────────
MAP_NAME   = 'Town10HD_Opt'
ANCHOR     = carla.Location(x=106.0, y=68.0, z=0.5)
LANE_WIDTH = 3.5  # m

# ── Blueprints ──────────────────────────────────────────────────────────────
BP_CONES = [
    'static.prop.constructioncone',
    'static.prop.trafficcone01',
    'static.prop.trafficcone02'
]
BP_BARRIERS = [
    'static.prop.streetbarrier',
]
BP_DEBRIS = [
    'static.prop.dirtdebris01',
    'static.prop.dirtdebris02',
    'static.prop.dirtdebris03'
]
BP_WARNING = 'static.prop.warningconstruction'
BP_SERVICE_VEHICLE = 'vehicle.mercedes.sprinter'

_POLICE_IDS = frozenset({
    'walker.pedestrian.0046',
    'walker.pedestrian.0047',
    'walker.pedestrian.0050',
    'walker.pedestrian.0051',
})

# Spectator settings
CAM_BACK, CAM_RIGHT, CAM_UP, CAM_PITCH = 40.0, -7.0, 18.0, -20.0

# ── Workzone Spawner ─────────────────────────────────────────────────────────

class WorkzoneSpawner:
    def __init__(self, world: carla.World) -> None:
        self.world = world
        self._map = world.get_map()
        self._lib = world.get_blueprint_library()
        self._actors: list[carla.Actor] = []
        self._traffic_manager = None

        self._anchor = self._map.get_waypoint(
            ANCHOR,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if self._anchor is None:
            raise RuntimeError(f"No drivable waypoint near anchor {ANCHOR}.")
            
    def set_traffic_manager(self, tm: carla.TrafficManager):
        self._traffic_manager = tm

    def _ahead(self, wp: carla.Waypoint, dist: float) -> carla.Waypoint:
        nxt = wp.next(dist)
        return nxt[0] if nxt else wp

    def _behind(self, wp: carla.Waypoint, dist: float) -> carla.Waypoint:
        prv = wp.previous(dist)
        return prv[0] if prv else wp

    def _right_offset(self, wp: carla.Waypoint, lateral: float, z_lift: float = 0.0) -> carla.Location:
        r = wp.transform.get_right_vector()
        base = wp.transform.location
        return carla.Location(
            x=base.x + r.x * lateral,
            y=base.y + r.y * lateral,
            z=base.z + z_lift,
        )

    def _spawn_with_jitter(self, bp_id: str, transform: carla.Transform, 
                         pos_jitter=0.15, rot_jitter=15.0) -> carla.Actor | None:
        bp = self._lib.find(bp_id)
        # Apply jitter
        transform.location.x += random.uniform(-pos_jitter, pos_jitter)
        transform.location.y += random.uniform(-pos_jitter, pos_jitter)
        transform.rotation.yaw += random.uniform(-rot_jitter, rot_jitter)
        
        actor = self.world.try_spawn_actor(bp, transform)
        if actor:
            self._actors.append(actor)
        return actor

    def build(self) -> None:
        LOG.info("Building realistic workzone...")
        
        # 1. Advance Warning Signs (10m - 50m upstream)
        for d in [50.0, 30.0, 15.0]:
            wp = self._behind(self._anchor, d)
            loc = self._right_offset(wp, LANE_WIDTH * 1.5)
            rot = carla.Rotation(yaw=wp.transform.rotation.yaw + 180.0)
            self._spawn_with_jitter(BP_WARNING, carla.Transform(loc, rot), 0.2, 5.0)

        # 2. Taper (Transition) - Diagonal Cones
        # L = (W * S^2) / 60 for 25mph -> approx 38m
        taper_len = 38.0
        num_cones = 8
        for i in range(num_cones):
            dist = (i / (num_cones - 1)) * taper_len
            wp = self._ahead(self._anchor, -taper_len + dist)
            lateral = (LANE_WIDTH / 2.0) + (1.0 - i / (num_cones - 1)) * LANE_WIDTH
            loc = self._right_offset(wp, lateral, z_lift=0.1)
            bp_cone = random.choice(BP_CONES)
            self._spawn_with_jitter(bp_cone, carla.Transform(loc, wp.transform.rotation))

        # 3. Buffer Zone & Service Vehicle
        buffer_len = 10.0
        wp_buffer = self._ahead(self._anchor, buffer_len / 2.0)
        loc_truck = self._right_offset(wp_buffer, LANE_WIDTH)
        truck_bp = self._lib.find(BP_SERVICE_VEHICLE)
        truck = self.world.try_spawn_actor(truck_bp, carla.Transform(loc_truck, wp_buffer.transform.rotation))
        if truck:
            self._actors.append(truck)
            truck.set_light_state(carla.VehicleLightState(carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker | carla.VehicleLightState.Position))
            LOG.info("Service vehicle spawned with blinkers.")

        # 4. Work Area (Barriers & Debris)
        work_len = 25.0
        num_barriers = 6
        for i in range(num_barriers):
            dist = buffer_len + (i * (work_len / num_barriers))
            wp = self._ahead(self._anchor, dist)
            loc = self._right_offset(wp, LANE_WIDTH / 2.0)
            self._spawn_with_jitter(random.choice(BP_BARRIERS), carla.Transform(loc, wp.transform.rotation), 0.05, 2.0)
            
            # Spawn debris randomly inside the work area
            if random.random() > 0.5:
                debris_loc = self._right_offset(wp, LANE_WIDTH * random.uniform(0.8, 1.2))
                self._spawn_with_jitter(random.choice(BP_DEBRIS), carla.Transform(debris_loc, wp.transform.rotation), 1.0, 180.0)

        # 5. Workers
        walker_bps = [bp for bp in self._lib.filter('walker.pedestrian.*') if bp.id in _POLICE_IDS]
        for _ in range(2):
            wp = self._ahead(self._anchor, buffer_len + random.uniform(5, 20))
            loc = self._right_offset(wp, LANE_WIDTH * 1.2)
            rot = carla.Rotation(yaw=random.uniform(0, 360))
            bp_walker = random.choice(walker_bps)
            self._spawn_with_jitter(bp_walker.id, carla.Transform(loc, rot), 0.5, 0)

        # 6. Termination Taper
        term_len = 15.0
        num_term_cones = 4
        work_end = buffer_len + work_len
        for i in range(num_term_cones):
            dist = work_end + (i / (num_term_cones - 1)) * term_len
            wp = self._ahead(self._anchor, dist)
            lateral = (LANE_WIDTH / 2.0) + (i / (num_term_cones - 1)) * LANE_WIDTH
            loc = self._right_offset(wp, lateral, z_lift=0.1)
            self._spawn_with_jitter(random.choice(BP_CONES), carla.Transform(loc, wp.transform.rotation))

        LOG.info("Workzone construction finished.")

    def cleanup(self) -> None:
        for actor in self._actors:
            if actor.is_alive:
                actor.destroy()
        self._actors = []
        LOG.info("Cleanup finished.")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    
    # Se o mapa não for o correto, carrega (opcional)
    if not world.get_map().name.endswith(MAP_NAME):
        world = client.load_world(MAP_NAME)
        time.sleep(2.0)

    spawner = WorkzoneSpawner(world)
    
    # Configurar Traffic Manager para evitar a faixa
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)
    # Exemplo: Se soubermos que a faixa da direita está fechada, podemos forçar offset
    # tm.global_percentage_lane_offset(-10.0) # Nudge para a esquerda

    try:
        spawner.build()
        
        # Posicionar espectador
        spec = world.get_spectator()
        wp = world.get_map().get_waypoint(ANCHOR)
        # Visão de cima/trás
        back_vec = wp.transform.get_forward_vector() * -20
        up_vec = carla.Location(z=15)
        spec.set_transform(carla.Transform(wp.transform.location + back_vec + up_vec, 
                                          carla.Rotation(pitch=-30, yaw=wp.transform.rotation.yaw)))
        
        LOG.info("Workzone deployed. Press Ctrl+C to cleanup and exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        spawner.cleanup()
    except Exception as e:
        LOG.error(f"Error: {e}")
        spawner.cleanup()

if __name__ == '__main__':
    main()
