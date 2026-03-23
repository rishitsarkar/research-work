import carla
import numpy as np
import time
import os
import queue
import threading
import random
import csv

# --- DATASET CONFIGURATION ---
TOTAL_SEQUENCES = 50
SNAPSHOTS_PER_SEQUENCE = 200
FRAMES_PER_SNAPSHOT = 2  # 2 frames at 20Hz = 100ms

BASE_DIR = "LiDAR_Dataset_Setting1"

# --- ATTACK CONFIGURATION ---
ATTACK_MIN_FRAMES = 4   # 400 ms
ATTACK_MAX_FRAMES = 16  # 1600 ms
ATTACK_DISTANCE = 20.0  # Meters ahead of the ego vehicle

def save_worker(save_queue, stop_event):
    """
    Background thread that saves both the .npy point clouds 
    and the labels.csv ground truth file simultaneously.
    """
    while not stop_event.is_set() or not save_queue.empty():
        try:
            seq_idx, snap_idx, data, is_spoofed, ghost_xyz = save_queue.get(timeout=1.0)
            
            seq_folder = os.path.join(BASE_DIR, f"Sequence_{seq_idx:02d}")
            os.makedirs(seq_folder, exist_ok=True)
            
            filename = f"snapshot_{snap_idx:03d}.npy"
            file_path = os.path.join(seq_folder, filename)
            np.save(file_path, data)
            
            csv_path = os.path.join(seq_folder, "labels.csv")
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if snap_idx == 1 and not file_exists:
                    writer.writerow(["Snapshot_File", "Status", "Ghost_X", "Ghost_Y", "Ghost_Z"])
                
                status = "Spoofed" if is_spoofed else "Clean"
                gx = f"{ghost_xyz[0]:.2f}" if ghost_xyz else "None"
                gy = f"{ghost_xyz[1]:.2f}" if ghost_xyz else "None"
                gz = f"{ghost_xyz[2]:.2f}" if ghost_xyz else "None"
                
                writer.writerow([filename, status, gx, gy, gz])
            
            save_queue.task_done()
        except queue.Empty:
            continue

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        print("Connected to CARLA. Syncing with ROS bridge...")
    except Exception as e:
        print(f"Could not connect to CARLA: {e}")
        return

    print("Waiting for world state to update...")
    world.wait_for_tick()
    world.wait_for_tick() 

    ego_tesla = None
    all_vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in all_vehicles:
        v_type = vehicle.type_id.lower()
        v_role = vehicle.attributes.get('role_name', 'none')
        if 'tesla' in v_type or v_role == 'ego_vehicle' or v_role == 'hero':
            ego_tesla = vehicle
            break

    if not ego_tesla:
        print("CRITICAL: Tesla not found! Ensure 'ros2 launch' is fully loaded.")
        return
    print(f"Target Acquired: {ego_tesla.type_id}")

    ghost_bp = bp_lib.find('vehicle.tesla.model3')
    if ghost_bp.has_attribute('color'):
        ghost_bp.set_attribute('color', '255,0,0')

    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '320000')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_transform = carla.Transform(carla.Location(x=0, z=2.4))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_tesla)

    save_queue = queue.Queue()
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=save_worker, args=(save_queue, stop_event))
    worker_thread.start()

    state = {
        'frame_buffer': [],
        'current_snapshot': 1,
        'current_sequence': 1,
        'finished': False,
        'attack_start': random.randint(30, 150),
        'attack_end': 0,
        'ghost_actor': None,
        'ghost_xyz': None
    }
    state['attack_end'] = state['attack_start'] + random.randint(ATTACK_MIN_FRAMES, ATTACK_MAX_FRAMES)

    def callback(data):
        if state['finished']:
            return

        raw_data = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(raw_data, (int(raw_data.shape[0] / 4), 4))
        xyz_points = points[:, :3].copy() 
        xyz_points[:, 1] = -xyz_points[:, 1]
        
        state['frame_buffer'].append(xyz_points)

        if len(state['frame_buffer']) == FRAMES_PER_SNAPSHOT:
            snapshot_data = np.concatenate(state['frame_buffer'], axis=0)
            
            is_spoofed = state['ghost_actor'] is not None
            
            save_queue.put((
                state['current_sequence'], 
                state['current_snapshot'], 
                snapshot_data, 
                is_spoofed, 
                state['ghost_xyz']
            ))
            
            state['frame_buffer'].clear()
            
            if state['current_snapshot'] % 50 == 0:
                print(f"Sequence {state['current_sequence']:02d} | Saved Snapshot {state['current_snapshot']:03d}/{SNAPSHOTS_PER_SEQUENCE}")
            
            state['current_snapshot'] += 1

            if state['current_snapshot'] > SNAPSHOTS_PER_SEQUENCE:
                print(f"--- Completed Sequence {state['current_sequence']:02d} ---")
                state['current_sequence'] += 1
                state['current_snapshot'] = 1
                
                if state['ghost_actor']:
                    state['ghost_actor'].destroy()
                    state['ghost_actor'] = None
                
                state['attack_start'] = random.randint(30, 150)
                state['attack_end'] = state['attack_start'] + random.randint(ATTACK_MIN_FRAMES, ATTACK_MAX_FRAMES)
                state['ghost_xyz'] = None

                if state['current_sequence'] > TOTAL_SEQUENCES:
                    state['finished'] = True
                    print("\nALL DATA COLLECTED! Shutting down gracefully...")

    lidar_sensor.listen(lambda data: callback(data))

    print(f"\nRECORDER ACTIVE: Gathering {TOTAL_SEQUENCES} adversarial sequences.")
    print("-------------------------------------------------------------------------")

    last_processed_snap = 0

    try:
        while not state['finished']:
            world.wait_for_tick()
            current_snap = state['current_snapshot']
            
            if current_snap != last_processed_snap:
                
                if current_snap == state['attack_start'] and state['ghost_actor'] is None:
                    ego_trans = ego_tesla.get_transform()
                    fw = ego_trans.get_forward_vector()
                    
                    spawn_loc = ego_trans.location + carla.Location(x=fw.x * ATTACK_DISTANCE, y=fw.y * ATTACK_DISTANCE, z=0.5)
                    spawn_trans = carla.Transform(spawn_loc, ego_trans.rotation)
                    
                    ghost = world.try_spawn_actor(ghost_bp, spawn_trans)
                    if ghost:
                        ghost.set_simulate_physics(False) # Freeze it in place
                        state['ghost_actor'] = ghost
                        state['ghost_xyz'] = (ATTACK_DISTANCE, 0.0, 0.0) 
                        print(f"  -> SPOOF DEPLOYED at Snapshot {current_snap} (Lasts until {state['attack_end']})")
                    else:
                        print(f"  -> WARNING: Spoof failed to spawn at Snapshot {current_snap} (Collision detected)")
                        state['attack_start'] += 1 
                        state['attack_end'] += 1

                elif current_snap == state['attack_end'] and state['ghost_actor'] is not None:
                    state['ghost_actor'].destroy()
                    state['ghost_actor'] = None
                    state['ghost_xyz'] = None
                    print(f"  -> SPOOF REMOVED at Snapshot {current_snap}")

                last_processed_snap = current_snap
                
    except KeyboardInterrupt:
        print("\nManual interrupt detected.")
    finally:
        if state['ghost_actor']:
            state['ghost_actor'].destroy()
        if lidar_sensor:
            lidar_sensor.stop()
            lidar_sensor.destroy()
            print("Lidar sensor detached.")
            
        print("Emptying remaining data to hard drive...")
        stop_event.set()
        worker_thread.join()
        print("Process complete.")

if __name__ == '__main__':
    main()
