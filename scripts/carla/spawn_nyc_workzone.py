#!/usr/bin/env python3
"""
spawn_nyc_workzone.py — ESV 2026 Work Zone Demo (Mi3 Lab)
Toronto, May 2026 | MUTCD Part 6 compliant

6 scenarios using only native CARLA 0.9.16 assets (no Fab imports):

  REAL WORK ZONES
  ─────────────────────────────────────────────────────────────────────
  wz01  Town04        Highway lane closure — CMS, W20-1, arrow board,
                      long taper, jersey barriers, 2 workers
  wz02  Town10HD_Opt  Urban active work — cone funnel, Type III barricades,
                      excavation area, 4 workers          [DEFAULT]
  wz03  Town10HD_Opt  Shoulder work only — subtle, 1 worker, service van,
                      no advance signs

  FALSE POSITIVES  (system must NOT classify as work zone)
  ─────────────────────────────────────────────────────────────────────
  fp01  Town04        Isolated cones on shoulder, no context
  fp02  Town10HD_Opt  Orange truck parked, no cones or workers
  fp03  Town10HD_Opt  Cluster of permanent signs + 1 casual cone

Usage:
    python spawn_nyc_workzone.py                  # wz02 (default)
    python spawn_nyc_workzone.py --scenario wz01
    python spawn_nyc_workzone.py --scenario fp02
    python spawn_nyc_workzone.py --cleanup        # purge all WZ props, exit
    python spawn_nyc_workzone.py --no-weather --no-spectator
"""

from __future__ import annotations

import argparse
import logging
import random
import signal
import sys
import time
from dataclasses import dataclass

import carla
from carla import VehicleLightState as vls

LOG = logging.getLogger('workzone')

# ──────────────────────────────────────────────────────────────────────────────
# Scenario Definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioDef:
    id: str
    name: str
    map_name: str
    anchor: carla.Location
    # Spectator camera offset relative to anchor (back, right, up, pitch_deg)
    cam: tuple[float, float, float, float]


SCENARIOS: dict[str, ScenarioDef] = {
    'wz01': ScenarioDef(
        id='wz01',
        name='Highway Lane Closure',
        map_name='Town04',
        # TOP straight westbound, outer driving lane (+5) — this is the lane being closed
        anchor=carla.Location(x=200.0, y=-8.0, z=0.5),
        cam=(-60.0, 12.0, 20.0, -22.0),
    ),
    'wz02': ScenarioDef(
        id='wz02',
        name='Urban Active Lane Closure',
        map_name='Town10HD_Opt',
        # Road 675: straight N-S section of the 4-lane boulevard, x≈106 (inner lane=-1).
        # Empirically confirmed: yaw≈-90° (southbound), get_right_vector()=(+1,0,0)=east.
        # Driver's right shoulder = east = POSITIVE lateral. y=68 confirmed straight. ✓
        anchor=carla.Location(x=106.0, y=68.0, z=0.5),
        cam=(40.0, -7.0, 18.0, -20.0),  # 40m north of anchor, looking south — shows taper
    ),
    'wz03': ScenarioDef(
        id='wz03',
        name='Shoulder Work (subtle)',
        map_name='Town10HD_Opt',
        # Road 675 straight section — shoulder work only, no lane closure.
        anchor=carla.Location(x=106.0, y=50.0, z=0.5),
        cam=(20.0, -6.0, 12.0, -22.0),
    ),
    'fp01': ScenarioDef(
        id='fp01',
        name='False Positive — Isolated cones',
        map_name='Town04',
        anchor=carla.Location(x=50.0, y=-12.0, z=0.5),   # TOP straight, westbound — separado do wz01
        cam=(-15.0, 5.0, 10.0, -22.0),
    ),
    'fp02': ScenarioDef(
        id='fp02',
        name='False Positive — Orange truck, no context',
        map_name='Town10HD_Opt',
        # Road 675 straight section — well away from real work-zone anchors.
        anchor=carla.Location(x=106.0, y=75.0, z=0.5),
        cam=(20.0, -6.0, 10.0, -20.0),
    ),
    'fp03': ScenarioDef(
        id='fp03',
        name='False Positive — Sign cluster + casual cone',
        map_name='Town10HD_Opt',
        anchor=carla.Location(x=106.0, y=78.0, z=0.5),
        cam=(20.0, -6.0, 10.0, -20.0),
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# MUTCD Layout Constants
# ──────────────────────────────────────────────────────────────────────────────

LANE_WIDTH = 3.5   # m  ≈ 12 ft typical urban lane

# Urban 25 mph — MUTCD Table 6H-3 "Lane Closure on Urban or Suburban Arterial"
# NYC DOT standard construction speed: 25 mph (NYC Admin. Code §19-182).
# Road 675 straight section: y=50..80 (30m). Anchor at y=68 → only 12m upstream
# before road 1 curves. sign_dists limited to 10m so signs stay on road 675.
# Downstream: road continues straight (roads 2, 566) → 95m available ✓
# Taper: L = WS²/60 = 12×25²/60 = 125 ft = 38 m (5 cones @ 7.5m ≤ 20ft max spacing)
# Buffer: 10 m (≥ 30ft min, Table 6C-2)  Work: 5 barriers × 4m = 20m  Term: 3 × 5m = 15m
# Total downstream: 37.5 + 10 + 20 + 15 = 82.5 m < 95 m available ✓
URBAN = dict(
    sign_dists    = (10.0,),          # 1 sign 10m upstream — stays on road 675 (ends at y=80)
    taper_cones   = 5,                # 5×7.5m = 37.5m ≈ 123ft ≈ MUTCD 125ft
    taper_spacing = 7.5,
    buffer        = 10.0,             # 33ft — above 30ft min (Table 6C-2)
    barriers      = 5,                # 5×4m = 20m work area
    barrier_gap   = 4.0,
    worker_offsets= (5.0, 13.0),
    term_cones    = 3,
    term_spacing  = 5.0,              # 3×5m = 15m termination taper
)

# Freeway 65 mph — MUTCD Table 6H-3 / Section 6C.07-6C.09
# Taper: L = W×S = 12ft × 65mph = 780ft = 238m  →  20 cones @ 12m
# Buffer: Table 6C-2 minimum 160ft (49m) → using 50m
# Tangent cone spacing: Table 6C-1 max 80ft (24m) at ≥45mph
HIGHWAY = dict(
    sign_dists      = (460.0, 310.0, 250.0, 155.0),  # W20-1, W4-2, R2-1, W20-5
    speed_sign_dist = 250.0,   # R2-1 "Speed Limit 45" — between 2nd and 3rd sign
    taper_cones     = 20,      # 20 cones × 12m = 240m ≈ 780ft (MUTCD L=WS)
    taper_spacing   = 12.0,    # max 40ft (12m) in taper (Table 6C-1)
    buffer          = 50.0,    # 160ft minimum (Table 6C-2, 65mph)
    barriers        = 20,      # jersey barriers continuous in work area
    barrier_gap     = 2.0,     # near-continuous placement
    tangent_cones   = 7,       # additional delineators in work area
    tangent_spacing = 24.0,    # max 80ft (24m) on tangent (Table 6C-1)
    worker_offsets  = (20.0, 55.0),
    term_cones      = 6,
    term_spacing    = 24.0,    # termination taper max 80ft (Table 6C-1)
)

# Lateral offsets — POSITIVE = driver's right (east for southbound road 675).
# Anchor on lane=-1 (x≈106). Lane=-2 (closed) center at x≈109.4 = +3.5m.
# Lane=-2 inner boundary = +1.75m. Right shoulder = +5.25m+.
OFF_SIGN_URBAN   = +(LANE_WIDTH * 2.0)   # +7.0m  beyond lane=-2, on sidewalk/shoulder
OFF_SIGN_HWY     = +(LANE_WIDTH + 1.5)   # +5.0m  wide shoulder (freeway)
OFF_BARRIER      = +(LANE_WIDTH / 2.0)   # +1.75m inner boundary of closed lane=-2
OFF_WORKER       = +(LANE_WIDTH + 0.5)   # +4.0m  inside closed lane=-2

# Native blueprint IDs (CARLA 0.9.16, no Fab assets needed)
# Mapped to YOLO detection classes (weights in jetson_config_defaults.yaml):
BP_CONE      = 'static.prop.constructioncone'  # → "Cone"      (channelization 0.9)
BP_CONE_ALT  = 'static.prop.trafficcone01'     # → "Cone"      (channelization 0.9, taller)
BP_DRUM      = 'static.prop.barrel'            # → "Drum"      (channelization 0.9)
BP_BARRICADE = 'static.prop.trafficwarning'    # → "Barricade" (channelization 0.9)
BP_BARRIER   = 'static.prop.streetbarrier'     # → "Barrier"   (channelization 0.9)
BP_FENCE     = 'static.prop.chainbarrier'      # → "Fence"     (channelization 0.9)
BP_WARNING   = 'static.prop.warningconstruction' # → TTC Sign  (signs 0.7)
BP_WARNING2  = 'static.prop.warningaccident'   # → TTC Sign    (signs 0.7)
BP_TRUCK     = 'vehicle.mercedes.sprinter'     # → "Work Vehicle" (vehicles 0.5)
BP_TRUCK_ALT = 'vehicle.volkswagen.t2'         # fallback — plain, no livery
                                               # NOTE: never use 'vehicle.volkswagen.*' wildcard
                                               # — t2_2021 has Coca-Cola livery, carlacola exists too

# Walker IDs confirmed as children in CARLA 0.9.16 — must never be used as construction workers
_CHILD_WALKERS = frozenset({
    'walker.pedestrian.0014',
    'walker.pedestrian.0021',
    'walker.pedestrian.0022',
    'walker.pedestrian.0023',
    'walker.pedestrian.0038',
})


# ──────────────────────────────────────────────────────────────────────────────
# WorkzoneSpawner
# ──────────────────────────────────────────────────────────────────────────────

class WorkzoneSpawner:
    """Builds any of the 6 ESV demo scenarios in CARLA."""

    def __init__(self, world: carla.World, scenario: ScenarioDef) -> None:
        self.world    = world
        self.scenario = scenario
        self._map     = world.get_map()
        self._lib     = world.get_blueprint_library()
        self._actors: list[carla.Actor] = []

        self._anchor = self._map.get_waypoint(
            scenario.anchor,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if self._anchor is None:
            raise RuntimeError(
                f"No drivable waypoint near anchor {scenario.anchor}. "
                f"Is {scenario.map_name} loaded?  "
                f"Run lane_explorer.py to find a valid coordinate."
            )
        LOG.info("Anchor → %s", self._anchor.transform.location)

    # ── road-graph helpers ────────────────────────────────────────────────────

    def _ahead(self, wp: carla.Waypoint, dist: float) -> carla.Waypoint:
        if dist <= 0.0:
            return wp
        nxt = wp.next(dist)
        return nxt[0] if nxt else wp

    def _behind(self, wp: carla.Waypoint, dist: float) -> carla.Waypoint:
        if dist <= 0.0:
            return wp
        prv = wp.previous(dist)
        return prv[0] if prv else wp

    def _right_offset(self, wp: carla.Waypoint, lateral: float,
                      z_lift: float = 0.15) -> carla.Location:
        r    = wp.transform.get_right_vector()
        base = wp.transform.location
        return carla.Location(
            x=base.x + r.x * lateral,
            y=base.y + r.y * lateral,
            z=base.z + z_lift,
        )

    # ── spawn helpers ─────────────────────────────────────────────────────────

    def _spawn(self, bp: carla.ActorBlueprint,
               loc: carla.Location, rot: carla.Rotation | None = None) -> carla.Actor | None:
        actor = self.world.try_spawn_actor(
            bp, carla.Transform(loc, rot or carla.Rotation())
        )
        if actor:
            self._actors.append(actor)
        return actor

    def _find(self, type_id: str) -> carla.ActorBlueprint | None:
        try:
            return self._lib.find(type_id)
        except Exception:
            LOG.warning("Blueprint not found: %s — phase skipped.", type_id)
            return None

    def _sign_rot(self, wp: carla.Waypoint) -> carla.Rotation:
        """Rotation for warning signs facing oncoming traffic.

        UE4/CARLA static props have their face on local +X. To make the sign
        face AGAINST traffic (toward approaching drivers) we need +180° so the
        face points upstream instead of downstream.
        """
        return carla.Rotation(yaw=wp.transform.rotation.yaw + 180.0)

    def _park_vehicle(self, type_id: str, color_rgb: str,
                      dist: float, lateral: float) -> carla.Actor | None:
        bp = self._find(type_id)
        if not bp:
            # Explicit fallback — avoid volkswagen.* wildcard (t2_2021 = Coca-Cola livery)
            bp = self._find(BP_TRUCK_ALT)
        if not bp:
            return None
        if bp.has_attribute('color'):
            try:
                bp.set_attribute('color', color_rgb)
            except Exception:
                pass
        wp  = self._ahead(self._anchor, dist)
        loc = self._right_offset(wp, lateral, z_lift=0.3)
        vehicle = self._spawn(bp, loc, wp.transform.rotation)
        if vehicle and isinstance(vehicle, carla.Vehicle):
            vehicle.apply_control(carla.VehicleControl(hand_brake=True))
            try:
                vehicle.set_light_state(
                    carla.VehicleLightState(vls.Special1 | vls.Special2)
                )
            except Exception:
                pass
        return vehicle

    # ── shared phases (used by multiple scenarios) ────────────────────────────

    def _place_advance_signs(self, dists: tuple[float, ...],
                             sign_offset: float) -> None:
        """Upstream warning signs on right shoulder, facing oncoming traffic."""
        bp = self._find(BP_WARNING)
        if not bp:
            return
        for dist in dists:
            wp  = self._behind(self._anchor, dist)
            loc = self._right_offset(wp, sign_offset)
            self._spawn(bp, loc, self._sign_rot(wp))
        LOG.info("  signs ✓  %d advance warning signs", len(dists))

    def _place_taper(self, n_cones: int, spacing: float) -> float:
        """Diagonal cone taper closing the right lane. Returns end dist.

        Sweeps from inner lane boundary outward to right shoulder, forcing
        traffic to merge left. Lateral range: +LANE_WIDTH/2 → +3*LANE_WIDTH/2
        (i.e., +1.75m → +5.25m for LANE_WIDTH=3.5m).
        """
        bp = self._find(BP_CONE)
        if not bp:
            return n_cones * spacing
        wp = self._anchor
        for i in range(n_cones):
            if i > 0:
                nxt = wp.next(spacing)
                wp  = nxt[0] if nxt else wp
            t       = i / max(n_cones - 1, 1)   # 0.0 → 1.0
            lateral = LANE_WIDTH / 2.0 + LANE_WIDTH * t  # +1.75 → +5.25
            self._spawn(bp, self._right_offset(wp, lateral), wp.transform.rotation)
        end = n_cones * spacing
        LOG.info("  taper ✓  %d cones (%.0f m) — inner boundary→shoulder", n_cones, end)
        return end

    def _place_barriers(self, start: float, count: int, gap: float) -> float:
        """Continuous barrier line along the work area. Returns end dist."""
        bp = self._find(BP_BARRIER)
        end = start + count * gap
        if not bp:
            return end
        for i in range(count):
            dist = start + i * gap
            wp   = self._ahead(self._anchor, dist)
            self._spawn(bp, self._right_offset(wp, OFF_BARRIER), wp.transform.rotation)
        LOG.info("  barriers ✓  %d units (%.0f m span)", count, end - start)
        return end

    def _adult_walkers(self) -> list[carla.ActorBlueprint]:
        """Return only adult walker blueprints, excluding known child IDs."""
        return [
            bp for bp in self._lib.filter('walker.pedestrian.*')
            if bp.id not in _CHILD_WALKERS
            and (not bp.has_attribute('age') or bp.get_attribute('age').as_str() == 'adult')
        ]

    # Police officer IDs confirmed in CARLA 0.9.16 pedestrian catalog (adult-sized).
    # 0048/0049 excluded — bounding box height=0.55m (child-sized in this build).
    _POLICE_IDS = frozenset({
        'walker.pedestrian.0046',
        'walker.pedestrian.0047',
        'walker.pedestrian.0050',
        'walker.pedestrian.0051',
    })

    def _police_walkers(self) -> list[carla.ActorBlueprint]:
        """Return police officer walker blueprints.

        Falls back to all adult walkers if none of the known police IDs are
        available on this CARLA server build.
        """
        candidates = [
            bp for bp in self._lib.filter('walker.pedestrian.*')
            if bp.id in self._POLICE_IDS
        ]
        if candidates:
            return candidates
        LOG.warning(
            "  workers ⚠  none of the known police IDs found (%s) — "
            "falling back to adult walkers. "
            "Run `python -c \"import carla; c=carla.Client(); "
            "print([b.id for b in c.get_world().get_blueprint_library()"
            ".filter('walker.pedestrian.*')])\"` to inspect available IDs.",
            ', '.join(sorted(self._POLICE_IDS)),
        )
        return self._adult_walkers()

    def _place_workers(self, work_start: float,
                       offsets: tuple[float, ...]) -> None:
        walker_bps = self._police_walkers()
        if not walker_bps:
            LOG.warning("  workers ✗  no walker blueprints found")
            return
        placed = 0
        for offset in offsets:
            wp  = self._ahead(self._anchor, work_start + offset)
            loc = self._right_offset(wp, OFF_WORKER)
            # Face toward oncoming traffic (upstream = +180° from road direction).
            rot = carla.Rotation(yaw=wp.transform.rotation.yaw + 180.0)
            if self._spawn(random.choice(walker_bps), loc, rot):
                placed += 1
        LOG.info("  workers ✓  %d police officers", placed)

    def _place_term_taper(self, work_end: float,
                          n_cones: int, spacing: float) -> None:
        """Reverse taper reopening the lane: shoulder → inner lane boundary."""
        bp = self._find(BP_CONE)
        if not bp:
            return
        wp = self._ahead(self._anchor, work_end)
        for i in range(n_cones):
            if i > 0:
                nxt = wp.next(spacing)
                wp  = nxt[0] if nxt else wp
            t       = i / max(n_cones - 1, 1)
            lateral = LANE_WIDTH / 2.0 + LANE_WIDTH * (1.0 - t)  # +5.25 → +1.75
            self._spawn(bp, self._right_offset(wp, lateral), wp.transform.rotation)
        LOG.info("  term ✓  %d cones — shoulder→inner boundary", n_cones)

    # ── scenario builders ─────────────────────────────────────────────────────

    def _build_wz01(self) -> None:
        """WZ-01  Town04 — Freeway right-lane closure at 65 mph.

        Full MUTCD Part 6 sequence (Table 6H-3, Section 6C.07-6C.09):

        ADVANCE WARNING AREA (upstream of taper):
          460m  W20-1  "Road Work Ahead"          — warningconstruction
          310m  W4-2   "Right Lane Closed Ahead"  — warningaccident (alt sign)
          250m  R2-1   "Speed Limit 45"           — warningconstruction (proxy)
          155m  W20-5  "Right Lane Closed"        — warningconstruction

        TRANSITION AREA:
            0m  Arrow board (Type C) — warningconstruction facing traffic
          Taper: L = W×S = 12ft × 65mph = 780ft = 238m
                 20 cones @ 12m spacing (MUTCD max 40ft in taper)

        BUFFER ZONE:  50m clear (MUTCD Table 6C-2: 160ft min at 65mph)

        WORK AREA:
          Jersey barriers @ 2m (near-continuous)
          Tangent cones @ 24m (MUTCD max 80ft on tangent)
          Service truck (orange) + 2 workers

        TERMINATION AREA:
          6 cones @ 24m reverse taper
          G20-2 "End Road Work" sign on right shoulder
        """
        p = HIGHWAY
        LOG.info("╔═ WZ-01  Freeway Lane Closure 65mph — Town04 ════════════════╗")

        # ── Advance Warning Area ──────────────────────────────────────────────
        # Sign 1: W20-1 "Road Work Ahead" at 1500ft (460m)
        bp1 = self._find(BP_WARNING)
        if bp1:
            wp  = self._behind(self._anchor, 460.0)
            self._spawn(bp1, self._right_offset(wp, OFF_SIGN_HWY), self._sign_rot(wp))

        # Sign 2: W4-2 "Right Lane Closed Ahead" at 1000ft (310m) — alt sign prop
        bp2 = self._find(BP_WARNING2) or self._find(BP_WARNING)
        if bp2:
            wp  = self._behind(self._anchor, 310.0)
            self._spawn(bp2, self._right_offset(wp, OFF_SIGN_HWY), self._sign_rot(wp))

        # Sign 3: R2-1 "Speed Limit 45" at 820ft (250m)
        bp3 = self._find(BP_WARNING)
        if bp3:
            wp  = self._behind(self._anchor, 250.0)
            self._spawn(bp3, self._right_offset(wp, OFF_SIGN_HWY), self._sign_rot(wp))

        # Sign 4: W20-5 "Right Lane Closed" at 500ft (155m)
        bp4 = self._find(BP_WARNING)
        if bp4:
            wp  = self._behind(self._anchor, 155.0)
            self._spawn(bp4, self._right_offset(wp, OFF_SIGN_HWY), self._sign_rot(wp))

        LOG.info("  signs ✓  W20-1 (460m) | W4-2 (310m) | R2-1 (250m) | W20-5 (155m)")

        # ── Arrow Board at taper start ────────────────────────────────────────
        # Type C flashing arrow panel — placed at right shoulder, faces traffic
        bp_ab = self._find(BP_WARNING)
        if bp_ab:
            wp  = self._anchor
            self._spawn(bp_ab, self._right_offset(wp, OFF_SIGN_HWY + 1.0),
                        self._sign_rot(wp))
            LOG.info("  arrow board ✓  taper start (Type C proxy)")

        # ── Transition Area — Cone Taper ──────────────────────────────────────
        # L = W×S = 12ft × 65mph = 780ft = 238m  →  20 cones @ 12m
        # Uses taller BP_CONE_ALT (28"+) where available (FHWA freeway requirement)
        bp_cone = self._find(BP_CONE_ALT) or self._find(BP_CONE)
        if bp_cone:
            wp_t = self._anchor
            for i in range(p['taper_cones']):
                if i > 0:
                    nxt = wp_t.next(p['taper_spacing'])
                    wp_t = nxt[0] if nxt else wp_t
                t       = i / max(p['taper_cones'] - 1, 1)
                lateral = LANE_WIDTH / 2.0 + LANE_WIDTH * t  # +1.75 → +5.25
                self._spawn(bp_cone, self._right_offset(wp_t, lateral),
                            wp_t.transform.rotation)
        taper_end = p['taper_cones'] * p['taper_spacing']
        LOG.info("  taper ✓  %d cones @ %.0fm = %.0fm (MUTCD L=WS=%.0fft)",
                 p['taper_cones'], p['taper_spacing'], taper_end,
                 taper_end * 3.281)

        # ── Buffer Zone ───────────────────────────────────────────────────────
        work_start = taper_end + p['buffer']
        LOG.info("  buffer ✓  %.0f m (%.0f ft) — clear zone", p['buffer'],
                 p['buffer'] * 3.281)

        # ── Work Area — Jersey Barriers ───────────────────────────────────────
        bp_bar = self._find(BP_BARRIER)
        if bp_bar:
            for i in range(p['barriers']):
                dist = work_start + i * p['barrier_gap']
                wp   = self._ahead(self._anchor, dist)
                self._spawn(bp_bar, self._right_offset(wp, OFF_BARRIER),
                            wp.transform.rotation)
        work_area_len = p['barriers'] * p['barrier_gap']
        work_end      = work_start + work_area_len
        LOG.info("  barriers ✓  %d jersey barriers (%.0f m span)", p['barriers'],
                 work_area_len)

        # ── Work Area — Tangent Delineator Cones (max 80ft/24m spacing) ───────
        if bp_cone:
            for i in range(p['tangent_cones']):
                dist = work_start + 30.0 + i * p['tangent_spacing']
                wp   = self._ahead(self._anchor, dist)
                self._spawn(bp_cone, self._right_offset(wp, OFF_BARRIER),
                            wp.transform.rotation)
            LOG.info("  tangent cones ✓  %d @ %.0fm spacing", p['tangent_cones'],
                     p['tangent_spacing'])

        # ── Work Area — Service Truck + Workers ───────────────────────────────
        mid = work_start + work_area_len * 0.35
        self._park_vehicle(BP_TRUCK, '255,140,0', mid, OFF_BARRIER + 2.5)
        LOG.info("  truck ✓  service vehicle (orange, warning lights)")

        self._place_workers(work_start, p['worker_offsets'])

        # ── Termination Area — Reverse Taper ─────────────────────────────────
        if bp_cone:
            wp_term = self._ahead(self._anchor, work_end)
            for i in range(p['term_cones']):
                if i > 0:
                    nxt = wp_term.next(p['term_spacing'])
                    wp_term = nxt[0] if nxt else wp_term
                t       = i / max(p['term_cones'] - 1, 1)
                lateral = LANE_WIDTH / 2.0 + LANE_WIDTH * (1.0 - t)  # +5.25 → +1.75
                self._spawn(bp_cone, self._right_offset(wp_term, lateral),
                            wp_term.transform.rotation)
            LOG.info("  term taper ✓  %d cones @ %.0fm", p['term_cones'],
                     p['term_spacing'])

        # G20-2 "End Road Work" sign — right shoulder past last cone
        bp_end = self._find(BP_WARNING)
        if bp_end:
            end_sign_dist = work_end + p['term_cones'] * p['term_spacing'] + 30.0
            wp  = self._ahead(self._anchor, end_sign_dist)
            self._spawn(bp_end, self._right_offset(wp, OFF_SIGN_HWY),
                        self._sign_rot(wp))
            LOG.info("  end sign ✓  G20-2 'End Road Work'")

        total_len = work_end + p['term_cones'] * p['term_spacing'] + 30.0
        LOG.info("╚═ WZ-01 complete: %d actors | total zone: %.0f m (%.0f ft) ════╝",
                 len(self._actors), total_len, total_len * 3.281)

    def _build_wz02(self) -> None:
        """WZ-02  Town10HD_Opt — Urban active lane closure (Road 10, 4-lane boulevard).

        NYC DOT / MUTCD Part 6 / Table 6H-3 "Lane Closure on Urban or Suburban Arterial"
        at 25 mph (NYC Admin. Code §19-182 mandates 25 mph in all work zones):

        ADVANCE WARNING AREA (right/east shoulder, upstream = north):
          W20-1 "Road Work Ahead"       200 ft / 61 m upstream
          R2-1  "Speed Limit 20"        150 ft / 46 m upstream
          W20-5 "Right Lane Closed"     100 ft / 30 m upstream   ← visible from spectator

        TRANSITION AREA (at anchor — taper start):
          Arrow board Type C (right shoulder)                     ← MUTCD 6F.58
          NYPD Traffic Enforcement Agent (faces oncoming traffic) ← NYC standard

        TAPER: L = WS²/60 = 12×25²/60 = 125 ft = 38 m  (7 cones @ 5.5 m, Table 6C-1)
               Cones sweep +1.75m → +5.25m (inner lane boundary → right shoulder)

        BUFFER ZONE: 12 m (40 ft, above 30 ft min, Table 6C-2)

        WORK AREA:
          6 jersey barriers × 3.5 m = 21 m                       ← fits Road 10 geometry
          Orange service van (center of work area)
          2 NYPD officers facing upstream

        TERMINATION AREA:
          3 cones @ 5 m reverse taper (MUTCD 6C.09)
          G20-2 "End Road Work" sign
        """
        p = URBAN
        LOG.info("╔═ WZ-02  NYC Urban Lane Closure 25 mph — Town10HD_Opt ════════╗")

        # ── Advance Warning Area ──────────────────────────────────────────────
        self._place_advance_signs(p['sign_dists'], OFF_SIGN_URBAN)

        # ── Transition Area: Arrow Board at taper start ───────────────────────
        # Type C flashing arrow panel (MUTCD 6F.58) — right shoulder, face upstream
        bp_ab = self._find(BP_WARNING2) or self._find(BP_WARNING)
        if bp_ab:
            loc = self._right_offset(self._anchor, OFF_SIGN_URBAN)
            self._spawn(bp_ab, loc, self._sign_rot(self._anchor))
            LOG.info("  arrow board ✓  Type C proxy at taper start")

        # NYPD TCO at taper entry — on right shoulder, facing upstream
        police_bps = self._police_walkers()
        if police_bps:
            loc = self._right_offset(self._anchor, OFF_SIGN_URBAN - 1.0)
            rot = carla.Rotation(yaw=self._anchor.transform.rotation.yaw + 180.0)
            self._spawn(random.choice(police_bps), loc, rot)
            LOG.info("  TCO ✓  NYPD officer at taper entry (facing upstream)")

        # ── Taper ─────────────────────────────────────────────────────────────
        taper_end  = self._place_taper(p['taper_cones'], p['taper_spacing'])

        # ── Buffer Zone ───────────────────────────────────────────────────────
        work_start = taper_end + p['buffer']
        LOG.info("  buffer ✓  %.0f m (%.0f ft)", p['buffer'], p['buffer'] * 3.281)

        # ── Work Area ─────────────────────────────────────────────────────────
        work_end = self._place_barriers(work_start, p['barriers'], p['barrier_gap'])

        # Service van parked on shoulder beyond closed lane
        mid = work_start + (work_end - work_start) * 0.4
        self._park_vehicle(BP_TRUCK, '255,140,0', mid, LANE_WIDTH + 1.5)
        LOG.info("  van ✓  service vehicle (orange, on shoulder)")

        self._place_workers(work_start, p['worker_offsets'])

        # ── Termination Area ──────────────────────────────────────────────────
        self._place_term_taper(work_end, p['term_cones'], p['term_spacing'])

        bp_end = self._find(BP_WARNING)
        if bp_end:
            end_dist = work_end + p['term_cones'] * p['term_spacing'] + 8.0
            wp_end   = self._ahead(self._anchor, end_dist)
            self._spawn(bp_end, self._right_offset(wp_end, OFF_SIGN_URBAN),
                        self._sign_rot(wp_end))
            LOG.info("  end sign ✓  G20-2 'End Road Work'")

        LOG.info("╚═ WZ-02 complete: %d actors ════════════════════════════════════╝",
                 len(self._actors))

    def _build_wz03(self) -> None:
        """WZ-03  Town10HD_Opt — Shoulder work only (subtle case).

        Layout: No advance signs. 6 cones along right shoulder edge only
                (no diagonal, no lane closure). 1 service van + 1 worker.
        """
        LOG.info("╔═ WZ-03  Shoulder Work (subtle) — Town10HD_Opt ═════════════╗")

        # Shoulder cones: straight line on right edge (no taper)
        bp = self._find(BP_CONE)
        if bp:
            for i in range(6):
                wp  = self._ahead(self._anchor, i * 4.0)
                loc = self._right_offset(wp, LANE_WIDTH + 0.2)
                self._spawn(bp, loc, wp.transform.rotation)
            LOG.info("  cones ✓  6 shoulder cones (no lane closure)")

        # Service van parked on shoulder
        self._park_vehicle(BP_TRUCK, '255,255,255', 12.0, LANE_WIDTH + 1.8)
        LOG.info("  van ✓  service van on shoulder")

        # Single worker (adult only)
        walker_bps = self._adult_walkers()
        if walker_bps:
            wp  = self._ahead(self._anchor, 8.0)
            loc = self._right_offset(wp, LANE_WIDTH + 0.8)
            rot = carla.Rotation(yaw=wp.transform.rotation.yaw)
            self._spawn(random.choice(walker_bps), loc, rot)
            LOG.info("  worker ✓  1 pedestrian")

        LOG.info("╚═ WZ-03 complete: %d actors ════════════════════════════════════╝",
                 len(self._actors))

    def _build_fp01(self) -> None:
        """FP-01  Town04 — Isolated cones on shoulder, no work zone context.

        4 cones scattered on right shoulder. No signs, no barriers, no workers.
        """
        LOG.info("╔═ FP-01  Isolated Cones (false positive) — Town04 ══════════╗")
        bp = self._find(BP_CONE)
        if bp:
            offsets = [(0.0, 0.3), (6.0, 0.5), (7.5, 0.2), (14.0, 0.4)]
            for dist, lat_extra in offsets:
                wp  = self._ahead(self._anchor, dist)
                loc = self._right_offset(wp, LANE_WIDTH + lat_extra)
                self._spawn(bp, loc, wp.transform.rotation)
            LOG.info("  cones ✓  4 isolated cones (no WZ context)")
        LOG.info("╚═ FP-01 complete: %d actors ════════════════════════════════════╝",
                 len(self._actors))

    def _build_fp02(self) -> None:
        """FP-02  Town10HD_Opt — Orange truck parked, no cones or workers."""
        LOG.info("╔═ FP-02  Orange Truck Only (false positive) — Town10HD_Opt ═╗")
        self._park_vehicle(BP_TRUCK, '255,140,0', 5.0, LANE_WIDTH + 1.5)
        LOG.info("  truck ✓  orange truck parked (no cones, no workers)")
        LOG.info("╚═ FP-02 complete: %d actors ════════════════════════════════════╝",
                 len(self._actors))

    def _build_fp03(self) -> None:
        """FP-03  Town10HD_Opt — Cluster of permanent signs + 1 casual cone."""
        LOG.info("╔═ FP-03  Sign Cluster + Casual Cone (false positive) — Town10HD_Opt ═╗")

        bp_warn = self._find(BP_WARNING)
        if bp_warn:
            # 3 signs grouped together (simulating a permanent sign cluster)
            cluster_positions = [(0.0, 0.0), (1.5, 0.3), (3.0, 0.0)]
            for dist, lat_extra in cluster_positions:
                wp  = self._ahead(self._anchor, dist)
                loc = self._right_offset(wp, OFF_SIGN_URBAN + lat_extra)
                self._spawn(bp_warn, loc, self._sign_rot(wp))
            LOG.info("  signs ✓  3 permanent-style signs (clustered)")

        # 1 cone nearby but not contextually related
        bp_cone = self._find(BP_CONE)
        if bp_cone:
            wp  = self._ahead(self._anchor, 8.0)
            loc = self._right_offset(wp, LANE_WIDTH * 0.3)
            self._spawn(bp_cone, loc, wp.transform.rotation)
            LOG.info("  cone ✓  1 casual cone (no WZ context)")

        LOG.info("╚═ FP-03 complete: %d actors ════════════════════════════════════╝",
                 len(self._actors))

    # ── public API ────────────────────────────────────────────────────────────

    def build(self) -> None:
        builders = {
            'wz01': self._build_wz01,
            'wz02': self._build_wz02,
            'wz03': self._build_wz03,
            'fp01': self._build_fp01,
            'fp02': self._build_fp02,
            'fp03': self._build_fp03,
        }
        builders[self.scenario.id]()

    def set_weather(self) -> None:
        is_highway = self.scenario.map_name == 'Town04'
        if is_highway:
            # Highway: bright clear day — high visibility, hard shadows
            params = carla.WeatherParameters(
                cloudiness=20.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wind_intensity=5.0,
                sun_azimuth_angle=200.0,
                sun_altitude_angle=50.0,
                fog_density=0.0,
                wetness=0.0,
            )
            LOG.info("Weather: clear highway day")
        else:
            # Urban: overcast NYC afternoon — diffused light, slightly damp
            params = carla.WeatherParameters(
                cloudiness=60.0,
                precipitation=0.0,
                precipitation_deposits=12.0,
                wind_intensity=10.0,
                sun_azimuth_angle=215.0,
                sun_altitude_angle=35.0,
                fog_density=0.0,
                wetness=15.0,
            )
            LOG.info("Weather: overcast NYC afternoon")
        self.world.set_weather(params)

    def position_spectator(self) -> None:
        back, right, up, pitch = self.scenario.cam
        fwd   = self._anchor.transform.get_forward_vector()
        rv    = self._anchor.transform.get_right_vector()
        base  = self._anchor.transform.location
        cam   = carla.Location(
            x=base.x + fwd.x * (-back) + rv.x * right,
            y=base.y + fwd.y * (-back) + rv.y * right,
            z=base.z + up,
        )
        self.world.get_spectator().set_transform(carla.Transform(
            cam,
            carla.Rotation(pitch=pitch, yaw=self._anchor.transform.rotation.yaw),
        ))
        LOG.info("Spectator positioned.")

    def cleanup(self) -> int:
        n = len(self._actors)
        for actor in self._actors:
            try:
                actor.destroy()
            except Exception:
                pass
        self._actors.clear()
        LOG.info("Cleanup: %d actors removed.", n)
        return n


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def purge_stale_actors(world: carla.World) -> int:
    """Destroy any leftover workzone props from a previous run."""
    WZ_KEYWORDS = ('cone', 'barrier', 'warningconstruction')
    stale = [
        a for a in world.get_actors().filter('static.prop.*')
        if any(kw in a.type_id for kw in WZ_KEYWORDS)
    ]
    for a in stale:
        try:
            a.destroy()
        except Exception:
            pass
    if stale:
        LOG.info("Purged %d stale workzone props.", len(stale))
    return len(stale)


def ensure_map(client: carla.Client, target: str) -> carla.World:
    world   = client.get_world()
    current = world.get_map().name.split('/')[-1]
    if current != target:
        LOG.info("Loading %s (current: %s) — please wait…", target, current)
        world = client.load_world(target)
        time.sleep(4.0)
    else:
        LOG.info("Map %s already loaded.", target)
    return world


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--scenario', default='wz02',
                    choices=list(SCENARIOS.keys()),
                    help='Scenario to spawn (default: wz02)')
    ap.add_argument('--host',    default='127.0.0.1', metavar='H')
    ap.add_argument('--port',    default=2000, type=int, metavar='P')
    ap.add_argument('--timeout', default=20.0, type=float)
    ap.add_argument('--cleanup', action='store_true',
                    help='Purge all workzone actors from current map and exit')
    ap.add_argument('--no-weather',   action='store_true')
    ap.add_argument('--no-spectator', action='store_true')
    ap.add_argument('--debug',        action='store_true')
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(levelname)s]  %(message)s',
        datefmt='%H:%M:%S',
    )

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    scenario = SCENARIOS[args.scenario]

    try:
        world = ensure_map(client, scenario.map_name)
    except Exception as exc:
        LOG.error("Connection/map failed: %s", exc)
        sys.exit(1)

    if args.cleanup:
        purge_stale_actors(world)
        LOG.info("Done. Exiting.")
        return

    LOG.info("Scenario: [%s] %s", scenario.id.upper(), scenario.name)

    spawner = WorkzoneSpawner(world, scenario)

    def _on_sigint(sig, frame):
        print()
        LOG.info("Interrupted — removing actors…")
        spawner.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_sigint)

    purge_stale_actors(world)

    if not args.no_weather:
        spawner.set_weather()

    spawner.build()

    if not args.no_spectator:
        spawner.position_spectator()

    LOG.info("Live. Ctrl+C to remove actors and exit.")
    signal.pause()


if __name__ == '__main__':
    main()
