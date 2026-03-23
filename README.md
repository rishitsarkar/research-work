# LiDAR Adversarial Dataset Pipeline

This repository contains a suite of Python scripts designed to interface with the CARLA Simulator. It is designed to generate ego-relative LiDAR point clouds and 3D bounding boxes for training autonomous driving perception models to detect False Data Injection (FDI) attacks.

## Repository Structure & File Overview

### Environment & Setup
* **`spawn_traffic.py`**
  Populates the CARLA map with rule-abiding background traffic. It interfaces directly with the CARLA Traffic Manager and syncs with the simulation clock. All spawned vehicles are forced to drive perfectly (no erratic lane changes, strict adherence to speed limits) to establish a controlled baseline environment.
* **`self_drive_tesla.py`**
  Acts as the autonomous controller for the Ego Vehicle. It handles the routing and driving logic for the target Tesla, allowing the data collection scripts to passively record the environment as the car navigates the map.

### Data Generators
* **`baseline_dataset_generator_traffic.py`**
  The clean data collector. Records the ego-vehicle driving through the traffic-populated environment without any adversarial attacks present. Background traffic bounding boxes are explicitly labeled as **Class 1** (`c = 1.0`).
* **`adversarial_generator_setting1.py`**
  Runs the core False Data Injection simulation in an isolated environment. Uses a dynamic timer to spawn and despawn a spoofed "ghost" vehicle directly in front of the ego vehicle to simulate a sudden sensor spoofing attack. 
* **`adversarial_generator_setting1_traffic.py`**
  The Master Attack Spawner. Combines the FDI attack logic with the background traffic environment. It records standard traffic as Class 1, and explicitly labels the spoofed ghost vehicle as **Class 2** (`c = 2.0`) so the perception model can learn to classify the anomaly.
* **`adversarial_generator_setting2.py`**
  Generates adversarial sequences under "Setting 2" parameters, altering the attack duration and deployment window to test the ML model's temporal sliding window memory.

### Visualization & Verification
* **`dataset_viewer_v2.py`**
  A custom Matplotlib 3D visualizer built to verify coordinate matrix transformations. It parses the dynamically generated datasets, renders the point clouds in a high-contrast White Mode, and draws mathematically exact 3D wireframe boxes around the vehicles. Standard traffic is boxed in **Blue**, while spoofed attacks are boxed in **Red** with an explicit `[STATUS: SPOOFED]` warning in the title.

---

## Data Format & Outputs

All generator scripts output synchronized folders containing the following structures for OpenPCDet ingestion:
* `snapshot_XXX.npy`: The ego-relative LiDAR point cloud for the 100ms sweep.
* `bboxes_XXX.npy`: An $N \times 10$ array containing the 3D bounding boxes for all vehicles within a 50-meter radius, mathematically transformed to match the ego-LiDAR coordinate frame: `[x, y, z, l, w, h, theta, vx, vy, class]`.
* `snapshot_XXX_ghost.npy`: (Adversarial scripts only) Isolated point cloud clusters representing the spoofed vehicle.
* `labels.csv`: A comprehensive index logging the file paths, attack status, and the Ego Vehicle's global position and absolute speed per frame.

## Execution Workflow

To ensure proper synchronization with the CARLA server, execute the scripts in the following order across separate terminal instances:

1. Launch the **CARLA Server**.
2. Run `python3 self_drive_tesla.py` to spawn and route the Ego Vehicle.
3. *(Optional)* Run `python3 spawn_traffic.py` to populate the map.
4. Run your chosen **Generator Script** (e.g., `python3 adversarial_generator_setting1_traffic.py`) to begin the collection loop.
5. Use `python3 dataset_viewer_v2.py` to inspect the generated arrays.
