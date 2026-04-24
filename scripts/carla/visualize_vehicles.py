#!/usr/bin/env python3
"""
visualize_vehicles.py - Spawns all vehicle blueprints in a grid for inspection.
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

    # Filtra todos os veículos e organiza por ID (que geralmente agrupa por marca/tipo)
    vehicle_blueprints = sorted(blueprint_library.filter('vehicle.*'), key=lambda bp: bp.id)

    print(f"--- CATALOGO DE VEICULOS ---")
    print(f"Encontrados {len(vehicle_blueprints)} veículos para visualizar.\n")

    # Posição inicial (Anchor)
    start_location = carla.Location(x=0, y=0, z=2.0)
    
    # Configuração da Grade (Veículos precisam de mais espaço: 8m x 12m)
    spacing_x = 12.0 # Distância frontal
    spacing_y = 8.0  # Distância lateral
    columns = 6
    
    spawned_actors = []

    try:
        for i, bp in enumerate(vehicle_blueprints):
            col = i % columns
            row = i // columns
            
            spawn_point = carla.Transform(
                carla.Location(
                    x = start_location.x + (row * spacing_x),
                    y = start_location.y + (col * spacing_y),
                    z = start_location.z
                ),
                carla.Rotation(yaw=0)
            )
            
            actor = world.try_spawn_actor(bp, spawn_point)
            if actor:
                spawned_actors.append(actor)
                print(f"[{i+1:3}/{len(vehicle_blueprints)}] Spawned: {bp.id}")
                
                # Opcional: Ativar luzes se for um veículo de emergência
                if 'police' in bp.id or 'fire' in bp.id or 'ambulance' in bp.id:
                    actor.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2))
            else:
                print(f"[{i+1:3}/{len(vehicle_blueprints)}] FAILED: {bp.id}")

        # Posiciona a câmera do espectador em uma visão aérea lateral
        spectator = world.get_spectator()
        spec_transform = carla.Transform(
            start_location + carla.Location(x=-20, y=20, z=30),
            carla.Rotation(pitch=-40, yaw=0)
        )
        spectator.set_transform(spec_transform)

        print("\n--- INSTRUCOES ---")
        print("1. Voe pelo mapa com WASD + Mouse.")
        print("2. Identifique o modelo e o ID no terminal.")
        print("3. Pressione Ctrl+C para limpar e sair.")
        
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nLimpando {len(spawned_actors)} veículos...")
        for actor in spawned_actors:
            actor.destroy()
        print("Saindo.")

if __name__ == '__main__':
    main()
