import os
import argparse
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2

from models.bisenet import BiSeNet
from utils.common import ATTRIBUTES, COLOR_LIST, letterbox, vis_parsing_maps

def prepare_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image_tensor = transform(image)
    image_batch = image_tensor.unsqueeze(0)

    return image_batch

def apply_overlay(original_image, mask, alpha=0.5):
    """
    Apply a transparent overlay of the segmentation mask on the original image.
    """
    # Convert images to numpy arrays
    image_np = np.array(original_image)
    mask_np = np.array(mask)

    # Define colors for the segmentation mask (Random colors for each label)
    colors = np.random.randint(0, 255, (np.max(mask_np) + 1, 3), dtype=np.uint8)

    # Apply color map
    color_mask = colors[mask_np]

    # Blend original image and mask
    overlay = cv2.addWeighted(image_np, 1 - alpha, color_mask, alpha, 0)

    return Image.fromarray(overlay)

@torch.no_grad()
def inference(config):
    output_path = config.output
    input_path = config.input
    weight = config.weight
    model = config.model

    output_path = os.path.join(output_path, model)
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19

    model = BiSeNet(num_classes, backbone_name=model)
    model.to(device)

    if os.path.exists(weight):
        model.load_state_dict(torch.load(weight))
    else:
        raise ValueError(f"Weights not found from given path ({weight})")

    if os.path.isfile(input_path):
        input_path = [input_path]

    model.eval()
    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        image = Image.open(file_path).convert("RGB")
        print(f"Processing image: {file_path}")

        # Store original image resolution
        original_size = image.size  # (width, height)

        # Resize to 512x512 for model input
        resized_image = image.resize((512, 512), resample=Image.BILINEAR)
        transformed_image = prepare_image(resized_image)
        image_batch = transformed_image.to(device)

        # Run inference
        output = model(image_batch)[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only
        predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

        # Convert mask to PIL Image for resizing
        mask_pil = Image.fromarray(predicted_mask.astype(np.uint8))

        # Resize mask back to original image resolution
        restored_mask = mask_pil.resize(original_size, resample=Image.NEAREST)

        # Save the resized segmentation mask
        # mask_save_path = os.path.join(output_path, f"{filename[:-4]}_mask.png")
        # restored_mask.save(mask_save_path)
        # print(f"Saved segmentation mask: {mask_save_path}")

        # Generate overlayed image
        overlayed_image = apply_overlay(image, restored_mask)

        # Save the overlayed image
        overlay_save_path = os.path.join(output_path, f"{filename[:-4]}_res.png")
        overlayed_image.save(overlay_save_path)
        print(f"Saved overlayed image: {overlay_save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Face parsing inference")
    parser.add_argument("--model", type=str, default="resnet18", help="model name, i.e resnet18, resnet34")
    parser.add_argument(
        "--weight",
        type=str,
        default="./weights/resnet18.pt",
        help="path to trained model, i.e resnet18/34"
    )
    parser.add_argument("--input", type=str, default="./assets/images/", help="path to an image or a folder of images")
    parser.add_argument("--output", type=str, default="./assets/", help="path to save model outputs")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inference(config=args)
