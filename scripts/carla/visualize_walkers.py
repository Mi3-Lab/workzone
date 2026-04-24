#!/usr/bin/env python3
"""
visualize_walkers.py - Spawns ONLY the last 5 pedestrian models for identification.
"""

import carla
import argparse
import time
import sys

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', default='127.0.0.1')
    argparser.add_argument('--port', type=int, default=2000)
    args = argparser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Filtra todos os pedestres e pega apenas os últimos 5
    all_walkers = sorted(blueprint_library.filter('walker.pedestrian.*'), key=lambda bp: bp.id)
    last_5_walkers = all_walkers[-5:]

    print(f"--- IDENTIFICADOR DE PERSONAGENS ---")
    print(f"Mostrando os últimos {len(last_5_walkers)} modelos da biblioteca.\n")

    # Posição inicial (Anchor)
    start_location = carla.Location(x=0, y=0, z=1.5)
    spacing = 3.0 
    
    spawned_actors = []

    try:
        for i, bp in enumerate(last_5_walkers):
            spawn_point = carla.Transform(
                carla.Location(
                    x = start_location.x,
                    y = start_location.y + (i * spacing),
                    z = start_location.z
                ),
                carla.Rotation(yaw=180) # De frente para você
            )
            
            actor = world.try_spawn_actor(bp, spawn_point)
            if actor:
                spawned_actors.append(actor)
                # PRINT MUITO CLARO DO ID NO TERMINAL
                print(f"PERSONAGEM NA POSICAO {i+1} (da esquerda para direita): {bp.id}")
            else:
                print(f"FALHA AO SPAWNAR: {bp.id}")

        # Posiciona a câmera do espectador bem de frente para os 5
        spectator = world.get_spectator()
        spec_transform = carla.Transform(
            start_location + carla.Location(x=-5, y=6, z=2),
            carla.Rotation(pitch=-10, yaw=0)
        )
        spectator.set_transform(spec_transform)

        print("\n--- INSTRUCOES ---")
        print("1. Olhe para os personagens no simulador.")
        print("2. Conte da esquerda para a direita (1, 2, 3, 4, 5).")
        print("3. Veja no terminal acima qual ID corresponde ao número que você gostou.")
        print("\nPressione Ctrl+C para sair.")
        
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        for actor in spawned_actors:
            actor.destroy()
        print("\nLimpo. Saindo.")

if __name__ == '__main__':
    main()
