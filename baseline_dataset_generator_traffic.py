import carla
import numpy as np
import time
import os
import queue
import threading
import math
import csv
from datetime import datetime

# --- DATASET CONFIGURATION ---
TOTAL_SEQUENCES = 50
SNAPSHOTS_PER_SEQUENCE = 200
FRAMES_PER_SNAPSHOT = 2  # 100ms sweeps

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = f"LiDAR_Dataset_Clean_OpenPCDet_{RUN_TIMESTAMP}"

def get_ego_relative_bbox(vehicle, lidar_transform, ego_velocity):
    """
    Calculates the 10-element bounding box array relative to the LiDAR sensor.
    Output: [x, y, z, l, w, h, theta, vx, vy, class]
    """
    # 1. Get exact 3D center of the bounding box in global coordinates
    bb_center = vehicle.get_transform().transform(vehicle.bounding_box.location)
    
    # 2. Calculate offset from LiDAR
    dx = bb_center.x - lidar_transform.location.x
    dy = bb_center.y - lidar_transform.location.y
    dz = bb_center.z - lidar_transform.location.z

    # 3. Rotate to match Ego heading (Inverse Rotation)
    yaw_rad = math.radians(-lidar_transform.rotation.yaw)
    rel_x = dx * math.cos(yaw_rad) - dy * math.sin(yaw_rad)
    rel_y = dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
    rel_z = dz

    # 4. Right-Handed Coordinate Flip (Matches the Point Cloud Y-flip)
    rel_y = -rel_y

    # 5. Dimensions (extent is half-length, so multiply by 2)
    extent = vehicle.bounding_box.extent
    l = extent.x * 2.0
    w = extent.y * 2.0
    h = extent.z * 2.0

    # 6. Relative Yaw (Theta)
    rel_yaw = math.radians(vehicle.get_transform().rotation.yaw - lidar_transform.rotation.yaw)
    rel_yaw = -rel_yaw # Flip for right-handed

    # 7. Relative Velocity
    v = vehicle.get_velocity()
    dvx = v.x - ego_velocity.x
    dvy = v.y - ego_velocity.y

    rel_vx = dvx * math.cos(yaw_rad) - dvy * math.sin(yaw_rad)
    rel_vy = dvx * math.sin(yaw_rad) + dvy * math.cos(yaw_rad)
    rel_vy = -rel_vy # Flip for right-handed

    c = 1.0 

    return [rel_x, rel_y, rel_z, l, w, h, rel_yaw, rel_vx, rel_vy, c]

def save_worker(save_queue, stop_event):
    while not stop_event.is_set() or not save_queue.empty():
        try:
            seq_idx, snap_idx, data, bboxes, ego_stats = save_queue.get(timeout=1.0)
            seq_folder = os.path.join(BASE_DIR, f"Sequence_{seq_idx:02d}")
            os.makedirs(seq_folder, exist_ok=True)
            
            filename = f"snapshot_{snap_idx:03d}.npy"
            np.save(os.path.join(seq_folder, filename), data)

            bbox_filename = f"bboxes_{snap_idx:03d}.npy"
            np.save(os.path.join(seq_folder, bbox_filename), bboxes)
            
            csv_path = os.path.join(seq_folder, "labels.csv")
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if snap_idx == 1 and not file_exists:
                    writer.writerow([
                        "Snapshot_File", "BBoxes_File", "Ego_Speed_ms", "Ego_X", "Ego_Y", "Ego_Z"
                    ])
                
                # ego_stats = (speed, x, y, z)
                writer.writerow([
                    filename, bbox_filename,
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
        print("Connected to CARLA. Syncing with ROS bridge...")
    except Exception as e:
        print(f"Could not connect to CARLA: {e}")
        return

    world.wait_for_tick()
    world.wait_for_tick() 

    # Find the Ego Tesla
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

    # Spawn LiDAR attached to the Tesla
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '320000')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_transform = carla.Transform(carla.Location(x=0, z=2.4))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_tesla)

    # Setup saving thread
    save_queue = queue.Queue()
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=save_worker, args=(save_queue, stop_event))
    worker_thread.start()

    state = {
        'frame_buffer': [],
        'current_snapshot': 1,
        'current_sequence': 1,
        'finished': False,
        'latest_bboxes': np.zeros((0, 10), dtype=np.float32),
        'latest_ego_stats': (0.0, 0.0, 0.0, 0.0)
    }

    def callback(data):
        if state['finished']:
            return

        # Parse Point Cloud
        raw_data = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(raw_data, (int(raw_data.shape[0] / 4), 4))
        xyz_points = points[:, :3].copy() 
        xyz_points[:, 1] = -xyz_points[:, 1]
        
        state['frame_buffer'].append(xyz_points)

        if len(state['frame_buffer']) == FRAMES_PER_SNAPSHOT:
            snapshot_data = np.concatenate(state['frame_buffer'], axis=0)

            save_queue.put((
                state['current_sequence'], 
                state['current_snapshot'], 
                snapshot_data, 
                state['latest_bboxes'],    # Pass the bounding boxes
                state['latest_ego_stats']  # Pass the ego metrics
            ))
            
            state['frame_buffer'].clear()
            
            if state['current_snapshot'] % 50 == 0:
                print(f"Sequence {state['current_sequence']:02d} | Saved Snapshot {state['current_snapshot']:03d}/{SNAPSHOTS_PER_SEQUENCE}")
            
            state['current_snapshot'] += 1

            if state['current_snapshot'] > SNAPSHOTS_PER_SEQUENCE:
                print(f"--- Completed Sequence {state['current_sequence']:02d} ---")
                state['current_sequence'] += 1
                state['current_snapshot'] = 1

                if state['current_sequence'] > TOTAL_SEQUENCES:
                    state['finished'] = True
                    print("\nALL DATA COLLECTED! Shutting down gracefully...")

    lidar_sensor.listen(lambda data: callback(data))

    print(f"\nRECORDER ACTIVE: Saving BASELINE to {BASE_DIR}")
    print("-------------------------------------------------------------------------")

    try:
        while not state['finished']:
            world.wait_for_tick()
            
            lidar_trans = lidar_sensor.get_transform()
            ego_vel = ego_tesla.get_velocity()
            ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
            ego_loc = ego_tesla.get_location()
            
            current_bboxes = []
            for v in world.get_actors().filter('vehicle.*'):
                if v.id == ego_tesla.id:
                    continue # Skip our own car
                    
                bbox = get_ego_relative_bbox(v, lidar_trans, ego_vel)
                
                dist = math.sqrt(bbox[0]**2 + bbox[1]**2 + bbox[2]**2)
                if dist <= 50.0:
                    current_bboxes.append(bbox)
            
            state['latest_bboxes'] = np.array(current_bboxes, dtype=np.float32) if current_bboxes else np.zeros((0, 10), dtype=np.float32)
            state['latest_ego_stats'] = (ego_speed, ego_loc.x, ego_loc.y, ego_loc.z)
                
    except KeyboardInterrupt:
        print("\nManual interrupt detected.")
    finally:
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
