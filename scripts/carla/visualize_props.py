#!/usr/bin/env python3
"""
visualize_props.py - Spawns all static.prop objects in a grid for inspection.
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

    # Filtra todos os props estáticos
    prop_blueprints = blueprint_library.filter('static.prop.*')
    prop_blueprints = sorted(prop_blueprints, key=lambda bp: bp.id)

    print(f"Encontrados {len(prop_blueprints)} objetos para visualizar.")

    # Posição inicial (Anchor)
    start_location = carla.Location(x=0, y=0, z=2.0)
    
    # Configuração da Grade
    spacing = 5.0 # metros entre objetos
    columns = 8
    
    spawned_actors = []

    try:
        for i, bp in enumerate(prop_blueprints):
            col = i % columns
            row = i // columns
            
            spawn_point = carla.Transform(
                carla.Location(
                    x = start_location.x + (row * spacing),
                    y = start_location.y + (col * spacing),
                    z = start_location.z
                ),
                carla.Rotation(yaw=0)
            )
            
            actor = world.try_spawn_actor(bp, spawn_point)
            if actor:
                spawned_actors.append(actor)
                print(f"[{i+1}/{len(prop_blueprints)}] Spawned: {bp.id}")
            else:
                print(f"[{i+1}/{len(prop_blueprints)}] FAILED: {bp.id}")

        # Posiciona a câmera do espectador para ver a grade
        spectator = world.get_spectator()
        spec_transform = carla.Transform(
            start_location + carla.Location(x=-10, y=15, z=20),
            carla.Rotation(pitch=-45, yaw=0)
        )
        spectator.set_transform(spec_transform)

        print("\nGrade completa! Use a camera do simulador (WASD + Mouse) para explorar.")
        print("Pressione Ctrl+C para remover todos os objetos e sair.")
        
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nLimpando {len(spawned_actors)} atores...")
        for actor in spawned_actors:
            actor.destroy()
        print("Saindo.")

if __name__ == '__main__':
    main()
