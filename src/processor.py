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


def slice_mask_grid(mask_grid_img):
    if len(mask_grid_img.shape) == 3:
        mask_grid_img = cv2.cvtColor(mask_grid_img, cv2.COLOR_RGB2GRAY)
    _, binary_grid = cv2.threshold(mask_grid_img, 127, 255, cv2.THRESH_BINARY)

    masks = []
    ps = 96
    for r in range(4):
        for c in range(4):
            masks.append(binary_grid[r * ps:(r + 1) * ps, c * ps:(c + 1) * ps])
    return masks


def calculate_dice(pred_binary, gt_mask):
    """
    Calculates Dice Score between two 2D binary masks.
    pred_binary: 2D array (0 or 255)
    gt_mask: 2D array (0 or 255)
    """
    # Ensure they are 2D (remove channel dimension if accidentally passed)
    if len(pred_binary.shape) == 3:
        pred_binary = pred_binary[:, :, 0]
    if len(gt_mask.shape) == 3:
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_RGB2GRAY)

    # Convert to 0/1 for math
    p = (pred_binary > 127).astype(np.float32)
    g = (gt_mask > 127).astype(np.float32)

    intersection = np.sum(p * g)
    total_area = np.sum(p) + np.sum(g)

    if total_area == 0:
        return 1.0  # Perfect match for two empty (normal) patches

    return (2. * intersection) / (total_area + 1e-8)

def get_tumor_percentage(mask):
    """Calculates the percentage of area covered by tumor cells."""
    tumor_pixels = np.sum(mask > 127)
    total_pixels = mask.size
    return (tumor_pixels / total_pixels) * 100


def get_binary_mask(heatmap_rgb):
    """
    Ensures Red (Tumor) becomes White (255)
    and Blue (Normal) becomes Black (0).
    """
    # In JET colormap, Red has high values in the Red channel
    # and Blue has low values in the Red channel.
    r_channel = heatmap_rgb[:, :, 0]

    # Thresholding: If red intensity is high (>127), it's Tumor (White)
    _, binary = cv2.threshold(r_channel, 127, 255, cv2.THRESH_BINARY)
    return binary

def get_gt_boundary_overlay(tissue_patch, gt_mask):
    """
    Draws a green boundary of the Ground Truth onto the tissue.
    """
    overlay = tissue_patch.copy()
    # Find edges in the ground truth mask
    contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours in Green (0, 255, 0) with thickness 2
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay