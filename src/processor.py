import numpy as np
import cv2
import torch


def process_grid_image(grid_img, model, alpha=0.4):
    """
    Slices a 384x384 grid (4x4 patches of 96x96)
    Upscales to 224x224 for the MobileNet-UNet
    Returns: Original Patches, Heatmaps, and Overlays
    """
    patch_size = 96
    target_size = 224  # Required by your trained model

    patches = []
    heatmaps = []
    overlays = []

    for r in range(4):
        for c in range(4):
            # 1. Extract the 96x96 patch
            y, x = r * patch_size, c * patch_size
            patch = grid_img[y:y + patch_size, x:x + patch_size]
            patches.append(patch)

            # 2. Upscale to 224x224 to match Training logic
            img_resized = cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

            # 3. Preprocess (To Tensor and Normalize)
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            # 4. Inference
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            # 5. Downscale probability map back to 96x96 for display
            prob_96 = cv2.resize(probs, (patch_size, patch_size))

            # 6. Create Color Heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * prob_96), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmaps.append(heatmap)

            # 7. Create Blended Overlay
            patch_overlay = cv2.addWeighted(patch, 1 - alpha, heatmap, alpha, 0)
            overlays.append(patch_overlay)

    return patches, heatmaps, overlays


def stitch_grid(patch_list):
    """Combines 16 patches into a single 4x4 image."""
    rows = [np.hstack(patch_list[i:i + 4]) for i in range(0, 16, 4)]
    return np.vstack(rows)