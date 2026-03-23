import carla
import random
import time
import logging

# --- TRAFFIC CONFIGURATION ---
NUM_VEHICLES = 100           # Moderate traffic density
TM_PORT = 8000              # Default Traffic Manager port
SAFE_DISTANCE = 3.0         # 3 meters of safe following distance

def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    
    vehicles_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(TM_PORT)
        
        traffic_manager.set_global_distance_to_leading_vehicle(SAFE_DISTANCE)
        traffic_manager.global_percentage_speed_difference(0.0) # Drive exactly at the speed limit
        
        settings = world.get_settings()
        if settings.synchronous_mode:
            traffic_manager.set_synchronous_mode(True)
            logging.info("Synchronous mode detected. Traffic Manager synced.")

        blueprints = world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        num_to_spawn = min(NUM_VEHICLES, number_of_spawn_points)
        
        if NUM_VEHICLES > number_of_spawn_points:
            logging.warning(f"Map only has {number_of_spawn_points} spawn points. Capping traffic at {num_to_spawn}.")
            
        random.shuffle(spawn_points)

        logging.info(f"Spawning {num_to_spawn} rule-abiding vehicles...")
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= num_to_spawn:
                break
            blueprint = random.choice(blueprints)
            
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        results = client.apply_batch_sync(batch, settings.synchronous_mode)
        
        for response in results:
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        for actor_id in vehicles_list:
            vehicle = world.get_actor(actor_id)
            if vehicle is not None:
                try:
                    traffic_manager.ignore_lights_percentage(vehicle, 0.0)
                    traffic_manager.ignore_signs_percentage(vehicle, 0.0)
                    traffic_manager.ignore_vehicles_percentage(vehicle, 0.0)
                    
                    traffic_manager.auto_lane_change(vehicle, False)
                    
                    traffic_manager.keep_right_rule_percentage(vehicle, 100.0)
                except Exception as e:
                    pass

        logging.info(f"Successfully spawned and configured {len(vehicles_list)} perfect drivers.")
        logging.info("Traffic is active. Press Ctrl+C to destroy traffic and exit.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("\nManual interrupt detected. Cleaning up traffic...")
    finally:
        logging.info(f"Destroying {len(vehicles_list)} vehicles...")
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        
        if world.get_settings().synchronous_mode:
            traffic_manager.set_synchronous_mode(False)
            
        logging.info("Traffic cleanup complete.")

if __name__ == '__main__':
    main()
