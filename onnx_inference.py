import onnxruntime as ort
import os
import cv2
import argparse
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from utils.common import ATTRIBUTES, COLOR_LIST, letterbox, vis_parsing_maps

# 🎨 Define Fixed Colors for Each Label
FIXED_COLORS = {
    0: (0, 0, 0, 0),         # Background (Transparent)
    1: (255, 0, 0, 128),     # Face (Red, Semi-Transparent)
    2: (0, 255, 0, 128),     # Eyes (Green, Semi-Transparent)
    3: (0, 0, 255, 128),     # Eyebrows (Blue, Semi-Transparent)
    4: (255, 255, 0, 128),   # Nose (Yellow, Semi-Transparent)
    5: (255, 165, 0, 128),   # Lips (Orange, Semi-Transparent)
    6: (128, 0, 128, 128),   # Hair (Purple, Semi-Transparent)
    7: (0, 255, 255, 128),   # Ears (Cyan, Semi-Transparent)
    8: (128, 128, 128, 128), # Neck (Gray, Semi-Transparent)
    9: (255, 20, 147, 128),  # Cheeks (Pink, Semi-Transparent)
}

def prepare_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image_tensor = transform(image)
    image_batch = image_tensor.unsqueeze(0)

    return image_batch.numpy()

def onnx_inference(onnx_session, image_batch):
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    output = onnx_session.run([output_name], {input_name: image_batch})[0]
    return output

def apply_overlay(original_image, mask, alpha=0.5):
    """
    Apply a transparent overlay of the segmentation mask on the original image.
    Uses predefined colors for consistency.
    """
    # Convert images to numpy arrays
    image_np = np.array(original_image).astype(np.uint8)
    mask_np = np.array(mask)

    # Create RGBA (4-channel) mask with transparency
    overlay_rgba = np.zeros((*mask_np.shape, 4), dtype=np.uint8)

    # Assign predefined colors
    for label, color in FIXED_COLORS.items():
        overlay_rgba[mask_np == label] = color  # Assign fixed color

    # Convert to PIL Image with transparency
    overlay_image = Image.fromarray(overlay_rgba, mode="RGBA")

    # Blend the overlay with the original image
    original_rgba = original_image.convert("RGBA")  # Convert original image to RGBA
    final_image = Image.alpha_composite(original_rgba, overlay_image)

    return final_image

def inference(config):
    output_path = config.output
    input_path = config.input
    weight = config.onnx_weight

    os.makedirs(output_path, exist_ok=True)

    onnx_session = ort.InferenceSession(weight)

    if os.path.isfile(input_path):
        input_path = [input_path]

    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        image = Image.open(file_path).convert("RGB")
        print(f"Processing image: {file_path}")

        # Store original image resolution
        original_size = image.size  # (width, height)

        # Resize to 512x512 for model input
        resized_image = image.resize((512, 512), resample=Image.BILINEAR)
        transformed_image = prepare_image(resized_image)

        # Run inference
        output = onnx_inference(onnx_session, transformed_image)
        predicted_mask = output.squeeze(0).argmax(0)

        # Convert mask to PIL Image for resizing
        mask_pil = Image.fromarray(predicted_mask.astype(np.uint8))

        # Resize mask back to original image resolution
        restored_mask = mask_pil.resize(original_size, resample=Image.NEAREST)

        # Save the resized segmentation mask
        mask_save_path = os.path.join(output_path, f"{filename[:-4]}_mask.png")
        restored_mask.save(mask_save_path)
        print(f"Saved segmentation mask: {mask_save_path}")

        # Generate overlayed image
        overlayed_image = apply_overlay(image, restored_mask)

        # Save the overlayed image
        overlay_save_path = os.path.join(output_path, f"{filename[:-4]}_res.png")
        overlayed_image.save(overlay_save_path)
        print(f"Saved overlayed image: {overlay_save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Face parsing ONNX inference")
    parser.add_argument(
        "--onnx-weight",
        type=str,
        default="./weights/resnet18.onnx",
        help="path to onnx model, default './weights/resnet18.onnx'"
    )
    parser.add_argument("--input", type=str, default="./assets/images/", help="path to an image or a folder of images")
    parser.add_argument("--output", type=str, default="./assets/", help="path to save model outputs")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inference(config=args)
