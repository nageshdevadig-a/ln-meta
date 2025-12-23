import cv2
import numpy as np
import os
from glob import glob


def create_sequential_grids(source_dir, save_dir, patch_size=96):
    # 1. Setup folders
    os.makedirs(save_dir, exist_ok=True)

    # 2. Get all patches and sort them to ensure consistent order
    patch_files = sorted(glob(os.path.join(source_dir, "*.png")))

    if len(patch_files) < 160:
        print(f"Warning: You have {len(patch_files)} images. 160 are needed for 10 grids.")

    # 3. Process in batches of 16
    for grid_idx in range(10):
        start = grid_idx * 16
        end = start + 16

        # Get the 16 paths for this specific grid
        batch_paths = patch_files[start:end]

        if len(batch_paths) < 16:
            print(f"Stopping: Not enough images left for Grid {grid_idx + 1}")
            break

        # Load images
        patches = [cv2.imread(p) for p in batch_paths]

        # 4. Create the 4x4 Grid Structure
        rows = []
        for i in range(0, 16, 4):
            # Horizontal stack 4 images for a row
            row = np.hstack(patches[i:i + 4])
            rows.append(row)

        # Vertical stack the 4 rows to make the 4x4 grid
        final_grid = np.vstack(rows)

        # 5. Save as PNG
        output_filename = f"global_test_grid_{grid_idx + 1}.png"
        output_path = os.path.join(save_dir, output_filename)
        cv2.imwrite(output_path, final_grid)

        print(f"✅ Created {output_filename} using images {start + 1} to {end}")


if __name__ == "__main__":
    # Update these paths to match your folder names
    SOURCE_PATH = "data/patches"
    SAVE_PATH = "data/test_grids"

    create_sequential_grids(SOURCE_PATH, SAVE_PATH)