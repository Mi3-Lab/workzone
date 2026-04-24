#!/usr/bin/env python3
"""
spawn_false_positives.py - Scenarios to test False Positive rejection.
1. Police stop (Vehicle only)
2. Minor maintenance (1 worker + 2 cones)
"""

import carla
import argparse
import random
import time

# --- Configurações dos Falsos Positivos (Áreas Distintas de Town10) ---
# PONTO A: Próximo à Rotatória Central (Cenário Urbano)
LOC_A = carla.Location(x=-20.0, y=-10.0, z=0.5) 
# PONTO B: Rua secundária perto da Orla/Porto
LOC_B = carla.Location(x=-80.0, y=120.0, z=0.5)

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', default='127.0.0.1')
    argparser.add_argument('--port', type=int, default=2000)
    args = argparser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    lib = world.get_blueprint_library()
    actors = []

    try:
        print("--- Gerando Falsos Positivos em Town10 ---")

        # --- CENÁRIO A: Abordagem Policial (Falso Positivo 1) ---
        wp_a = world.get_map().get_waypoint(LOC_A, project_to_road=True)
        # Viatura de polícia no acostamento
        bp_police = lib.find('vehicle.dodge.charger_police')
        trans_a = carla.Transform(
            wp_a.transform.location + wp_a.transform.get_right_vector() * 4.5,
            wp_a.transform.rotation
        )
        police = world.try_spawn_actor(bp_police, trans_a)
        if police:
            police.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2))
            actors.append(police)
            print(f"[FP-1] Viatura de polícia spawnada em {wp_a.transform.location}")

        # --- CENÁRIO B: Manutenção Simples (Falso Positivo 2) ---
        wp_b = world.get_map().get_waypoint(LOC_B, project_to_road=True)
        # Um trabalhador e apenas dois cones (não bloqueia a via)
        bp_worker = lib.find('walker.pedestrian.0052')
        bp_cone = lib.find('static.prop.constructioncone')
        
        # Spawn do Trabalhador
        trans_b = carla.Transform(
            wp_b.transform.location + wp_b.transform.get_right_vector() * 3.5,
            wp_b.transform.rotation
        )
        worker = world.try_spawn_actor(bp_worker, trans_b)
        if worker:
            actors.append(worker)
            # Dois cones ao lado dele
            world.spawn_actor(bp_cone, carla.Transform(trans_b.location + carla.Location(x=1.5)))
            world.spawn_actor(bp_cone, carla.Transform(trans_b.location + carla.Location(x=-1.5)))
            print(f"[FP-2] Manutenção simples (1 trabalhador + 2 cones) spawnada em {wp_b.transform.location}")

        print("\nFalsos positivos ativos. O seu sistema deve ignorar estes pontos.")
        print("Pressione Ctrl+C para remover os atores e sair.")
        
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nLimpando {len(actors)} atores de falsos positivos...")
        for a in actors:
            if a.is_alive:
                a.destroy()
        print("Saindo.")

if __name__ == '__main__':
    main()
