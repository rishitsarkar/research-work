import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def select_dataset():
    """Scans the current directory for dataset folders and lets the user choose."""
    available_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('LiDAR_Dataset')]
    
    if not available_dirs:
        print("ERROR: No dataset folders found in the current directory.")
        return None
        
    print("\n=== Available Datasets ===")
    # Sort them so the newest timestamps appear at the bottom
    available_dirs.sort() 
    for i, d in enumerate(available_dirs):
        print(f"[{i+1}] {d}")
        
    while True:
        choice = input(f"\nSelect a dataset to load (1-{len(available_dirs)}) or 'q' to quit: ")
        if choice.lower() == 'q':
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_dirs):
                return available_dirs[idx]
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def view_frame():
    print("\n=== LiDAR Adversarial Dataset Viewer ===")
    
    BASE_DIR = select_dataset()
    if not BASE_DIR:
        print("Exiting viewer.")
        return
        
    print(f"\nTargeting directory: {BASE_DIR}")
    print("Type 'q' at any prompt to return to the main menu.")
    
    while True:
        seq_input = input("\nEnter Sequence Number (1-50): ")
        if seq_input.lower() == 'q':
            break
            
        snap_input = input("Enter Snapshot Number (1-200): ")
        if snap_input.lower() == 'q':
            break
            
        try:
            seq_num = int(seq_input)
            snap_num = int(snap_input)
        except ValueError:
            print("Invalid input. Please enter numbers only.")
            continue

        seq_folder = f"Sequence_{seq_num:02d}"
        snap_file = f"snapshot_{snap_num:03d}.npy"
        file_path = os.path.join(BASE_DIR, seq_folder, snap_file)
        csv_path = os.path.join(BASE_DIR, seq_folder, "labels.csv")

        if not os.path.exists(file_path):
            print(f"ERROR: Could not find file at -> {file_path}")
            continue

        # Parse the labels.csv to check attack status and get the ghost file
        is_spoofed = False
        ghost_coords = None
        ghost_file = "None"
        
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Snapshot_File'] == snap_file:
                        if row['Status'] == 'Spoofed':
                            is_spoofed = True
                            ghost_coords = (float(row['Ghost_Center_X']), float(row['Ghost_Center_Y']), float(row['Ghost_Center_Z']))
                            # Use .get() to avoid crashing on older datasets that lack this column
                            ghost_file = row.get('Ghost_Points_File', 'None')
                        break
        else:
            print("Warning: labels.csv not found for this sequence.")

        print(f"Loading {file_path}...")
        main_data = np.load(file_path)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the main environment (downsampled to prevent AnyDesk lag)
        ax.scatter(main_data[::5, 0], main_data[::5, 1], main_data[::5, 2], 
                   s=0.5, c=main_data[::5, 2], cmap='viridis', alpha=0.5)

        # Plot the exact points of the spoofed vehicle
        if is_spoofed:
            if ghost_file != "None":
                ghost_path = os.path.join(BASE_DIR, seq_folder, ghost_file)
                if os.path.exists(ghost_path):
                    ghost_data = np.load(ghost_path)
                    print(f"Loaded {len(ghost_data)} isolated ghost points.")
                    
                    # Plot the ghost points in solid bright red, slightly larger
                    ax.scatter(ghost_data[:, 0], ghost_data[:, 1], ghost_data[:, 2], 
                               color='red', s=5, label='Spoofed Vehicle Points')
                else:
                    print(f"Warning: Could not find ghost point file -> {ghost_file}")

            # Add floating text over the attack location
            if ghost_coords:
                gx, gy, gz = ghost_coords
                ax.text(gx, gy, gz + 4, "SPOOFED ATTACK", color='red', fontsize=12, fontweight='bold', ha='center')
            
            ax.legend()
            title_status = "[STATUS: SPOOFED ATTACK FRAME]"
        else:
            title_status = "[STATUS: CLEAN FRAME]"

        # Lock the axes to the 50m LiDAR range
        ax.set_xlim3d([-50, 50])
        ax.set_ylim3d([-50, 50])
        ax.set_zlim3d([-5, 20])

        ax.set_xlabel('X (Forward)')
        ax.set_ylabel('Y (Right)')
        ax.set_zlabel('Z (Up)')
        ax.set_title(f"Tesla Ego-Cloud | Sequence {seq_num:02d} - Snapshot {snap_num:03d}\n{title_status}")

        print("Opening plot window. Close the window to load another frame.")
        plt.show()

if __name__ == '__main__':
    view_frame()
