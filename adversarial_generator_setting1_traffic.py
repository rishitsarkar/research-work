import carla
import numpy as np
import time
import os
import queue
import threading
import random
import csv
import math
from datetime import datetime

# --- DATASET CONFIGURATION ---
TOTAL_SEQUENCES = 50
SNAPSHOTS_PER_SEQUENCE = 200
FRAMES_PER_SNAPSHOT = 2  # 100ms sweeps

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = f"LiDAR_Dataset_Adv_OpenPCDet_{RUN_TIMESTAMP}"

# --- ATTACK CONFIGURATION ---
ATTACK_MIN_FRAMES = 4    # 400 ms
ATTACK_MAX_FRAMES = 16   # 1600 ms
ATTACK_DISTANCE = 20.0   # Meters ahead of the ego vehicle

def get_ego_relative_bbox(vehicle, lidar_transform, ego_velocity, is_ghost=False):
    """
    Calculates the 10-element bounding box array relative to the LiDAR sensor.
    Output: [x, y, z, l, w, h, theta, vx, vy, class]
    """
    bb_center = vehicle.get_transform().transform(vehicle.bounding_box.location)
    
    dx = bb_center.x - lidar_transform.location.x
    dy = bb_center.y - lidar_transform.location.y
    dz = bb_center.z - lidar_transform.location.z

    yaw_rad = math.radians(-lidar_transform.rotation.yaw)
    rel_x = dx * math.cos(yaw_rad) - dy * math.sin(yaw_rad)
    rel_y = dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
    rel_z = dz

    rel_y = -rel_y # Right-Handed Coordinate Flip

    extent = vehicle.bounding_box.extent
    l = extent.x * 2.0
    w = extent.y * 2.0
    h = extent.z * 2.0

    rel_yaw = math.radians(vehicle.get_transform().rotation.yaw - lidar_transform.rotation.yaw)
    rel_yaw = -rel_yaw 

    v = vehicle.get_velocity()
    dvx = v.x - ego_velocity.x
    dvy = v.y - ego_velocity.y

    rel_vx = dvx * math.cos(yaw_rad) - dvy * math.sin(yaw_rad)
    rel_vy = dvx * math.sin(yaw_rad) + dvy * math.cos(yaw_rad)
    rel_vy = -rel_vy 

    # 1.0 for Background Traffic, 2.0 for Spoofed Ghost
    c = 2.0 if is_ghost else 1.0

    return [rel_x, rel_y, rel_z, l, w, h, rel_yaw, rel_vx, rel_vy, c]

def save_worker(save_queue, stop_event):
    while not stop_event.is_set() or not save_queue.empty():
        try:
            seq_idx, snap_idx, data, is_spoofed, ghost_xyz, ghost_points, bboxes, ego_stats = save_queue.get(timeout=1.0)
            seq_folder = os.path.join(BASE_DIR, f"Sequence_{seq_idx:02d}")
            os.makedirs(seq_folder, exist_ok=True)
            
            # Save Full Point Cloud
            filename = f"snapshot_{snap_idx:03d}.npy"
            np.save(os.path.join(seq_folder, filename), data)

            # Save Bounding Boxes
            bbox_filename = f"bboxes_{snap_idx:03d}.npy"
            np.save(os.path.join(seq_folder, bbox_filename), bboxes)

            # Save Ghost Points (If active)
            ghost_filename = "None"
            if is_spoofed and ghost_points is not None and len(ghost_points) > 0:
                ghost_filename = f"snapshot_{snap_idx:03d}_ghost.npy"
                np.save(os.path.join(seq_folder, ghost_filename), ghost_points)
            
            # Append to labels.csv
            csv_path = os.path.join(seq_folder, "labels.csv")
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if snap_idx == 1 and not file_exists:
                    writer.writerow([
                        "Snapshot_File", "Status", "Ghost_Center_X", "Ghost_Center_Y", "Ghost_Center_Z", 
                        "Ghost_Points_File", "BBoxes_File", "Ego_Speed_ms", "Ego_X", "Ego_Y", "Ego_Z"
                    ])
                
                status = "Spoofed" if is_spoofed else "Clean"
                gx = f"{ghost_xyz[0]:.2f}" if ghost_xyz else "None"
                gy = f"{ghost_xyz[1]:.2f}" if ghost_xyz else "None"
                gz = f"{ghost_xyz[2]:.2f}" if ghost_xyz else "None"
                
                writer.writerow([
                    filename, status, gx, gy, gz, ghost_filename, bbox_filename,
                    f"{ego_stats[0]:.2f}", f"{ego_stats[1]:.2f}", f"{ego_stats[2]:.2f}", f"{ego_stats[3]:.2f}"
                ])
            
            save_queue.task_done()
        except queue.Empty:
            continue

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        print("Connected to CARLA.")
    except Exception as e:
        print(f"Could not connect: {e}")
        return

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
        print("CRITICAL: Tesla not found.")
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
        'ghost_xyz': None,
        'latest_bboxes': np.zeros((0, 10), dtype=np.float32),
        'latest_ego_stats': (0.0, 0.0, 0.0, 0.0)
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
            ghost_points = None

            if is_spoofed:
                # The geocage mask extracts points around the static ATTACK_DISTANCE spawn
                x_min, x_max = ATTACK_DISTANCE - 2.5, ATTACK_DISTANCE + 2.5
                y_min, y_max = -1.5, 1.5
                z_min, z_max = -2.5, 0.5 

                mask = (snapshot_data[:, 0] > x_min) & (snapshot_data[:, 0] < x_max) & \
                       (snapshot_data[:, 1] > y_min) & (snapshot_data[:, 1] < y_max) & \
                       (snapshot_data[:, 2] > z_min) & (snapshot_data[:, 2] < z_max)
                
                ghost_points = snapshot_data[mask]

            # Send payload to background saving thread
            save_queue.put((
                state['current_sequence'], 
                state['current_snapshot'], 
                snapshot_data, 
                is_spoofed, 
                state['ghost_xyz'],
                ghost_points,
                state['latest_bboxes'],
                state['latest_ego_stats']
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

    print(f"\nRECORDER ACTIVE: Saving ADVERSARIAL to {BASE_DIR}")
    print("-------------------------------------------------------------------------")

    last_processed_snap = 0

    try:
        while not state['finished']:
            world.wait_for_tick()
            
            # --- EXTRACT GROUND TRUTH BBOXES ---
            lidar_trans = lidar_sensor.get_transform()
            ego_vel = ego_tesla.get_velocity()
            ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
            ego_loc = ego_tesla.get_location()
            
            current_bboxes = []
            for v in world.get_actors().filter('vehicle.*'):
                if v.id == ego_tesla.id:
                    continue
                    
                is_ghost = (state['ghost_actor'] is not None) and (v.id == state['ghost_actor'].id)
                bbox = get_ego_relative_bbox(v, lidar_trans, ego_vel, is_ghost)
                
                # Filter out vehicles beyond 50m LiDAR range
                dist = math.sqrt(bbox[0]**2 + bbox[1]**2 + bbox[2]**2)
                if dist <= 50.0:
                    current_bboxes.append(bbox)
            
            state['latest_bboxes'] = np.array(current_bboxes, dtype=np.float32) if current_bboxes else np.zeros((0, 10), dtype=np.float32)
            state['latest_ego_stats'] = (ego_speed, ego_loc.x, ego_loc.y, ego_loc.z)

            # --- SPOOF ATTACK TIMING ---
            current_snap = state['current_snapshot']
            if current_snap != last_processed_snap:
                if current_snap == state['attack_start'] and state['ghost_actor'] is None:
                    ego_trans = ego_tesla.get_transform()
                    fw = ego_trans.get_forward_vector()
                    
                    spawn_loc = ego_trans.location + carla.Location(x=fw.x * ATTACK_DISTANCE, y=fw.y * ATTACK_DISTANCE, z=0.5)
                    spawn_trans = carla.Transform(spawn_loc, ego_trans.rotation)
                    
                    ghost = world.try_spawn_actor(ghost_bp, spawn_trans)
                    if ghost:
                        ghost.set_simulate_physics(False) 
                        state['ghost_actor'] = ghost
                        state['ghost_xyz'] = (ATTACK_DISTANCE, 0.0, 0.0) 
                        print(f"  -> SPOOF DEPLOYED at Snapshot {current_snap}")
                    else:
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
            
        print("Emptying remaining data to hard drive...")
        stop_event.set()
        worker_thread.join()
        print("Process complete.")

if __name__ == '__main__':
    main()