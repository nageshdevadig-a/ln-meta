import cv2
import numpy as np
import os
from glob import glob


def create_sequential_grids(source_dir, save_dir, patch_size=96):
    # 1. Setup folders
    os.makedirs(save_dir, exist_ok=True)

    # 2. Get all patches_v1 and sort them to ensure consistent order
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


def create_synchronized_grids(parent_dir, save_parent_dir, patch_size=96):
    # 1. Setup Source and Save folders
    source_img_dir = os.path.join(parent_dir, "images")
    source_mask_dir = os.path.join(parent_dir, "masks")

    save_img_grid_dir = os.path.join(save_parent_dir, "image_grids")
    save_mask_grid_dir = os.path.join(save_parent_dir, "mask_grids")

    os.makedirs(save_img_grid_dir, exist_ok=True)
    os.makedirs(save_mask_grid_dir, exist_ok=True)

    # 2. Get and sort files (identical names ensure they match)
    img_files = sorted(glob(os.path.join(source_img_dir, "*.png")))
    mask_files = sorted(glob(os.path.join(source_mask_dir, "*.png")))

    total_available = min(len(img_files), len(mask_files))
    num_grids = total_available // 16

    print(f"Found {total_available} pairs. Creating {num_grids} grids...")

    # 3. Process in batches of 16
    for grid_idx in range(num_grids):
        start = grid_idx * 16
        end = start + 16

        batch_img_paths = img_files[start:end]
        batch_mask_paths = mask_files[start:end]

        # Load images and masks
        img_patches = [cv2.imread(p) for p in batch_img_paths]
        mask_patches = [cv2.imread(p) for p in batch_mask_paths]

        # 4. Create the 4x4 Grid Structure for both
        def build_grid(patch_list):
            rows = []
            for i in range(0, 16, 4):
                row = np.hstack(patch_list[i:i + 4])
                rows.append(row)
            return np.vstack(rows)

        final_img_grid = build_grid(img_patches)
        final_mask_grid = build_grid(mask_patches)

        # 5. Save Grids
        grid_filename = f"grid_{grid_idx + 1:02d}.png"

        cv2.imwrite(os.path.join(save_img_grid_dir, grid_filename), final_img_grid)
        cv2.imwrite(os.path.join(save_mask_grid_dir, grid_filename), final_mask_grid)

        print(f"✅ Created Grid {grid_idx + 1}: {grid_filename} (Synced Image & Mask)")

if __name__ == "__main__":
    # Update these paths to match your folder names
    SOURCE_PATH = "data/patches_v2"
    SAVE_PATH = "data/test_grids_v2"

    '''Use this function when you want to generate grids from a single set of images.'''
    # create_sequential_grids(SOURCE_PATH, SAVE_PATH)

    '''Use this function when you have synchronized images and masks in separate folders.'''
    create_synchronized_grids(SOURCE_PATH, SAVE_PATH)