import os
import argparse
import logging
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms

from models.bisenet import BiSeNet
from utils.common import vis_parsing_maps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def prepare_image(image: Image.Image, input_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    """
    Prepare an image for inference by resizing and normalizing it.

    Args:
        image: PIL Image to process
        input_size: Target size for resizing

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    # Resize the image
    resized_image = image.resize(input_size, resample=Image.BILINEAR)

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Apply transformations
    image_tensor = transform(resized_image)
    image_batch = image_tensor.unsqueeze(0)

    return image_batch


def load_model(model_name: str, num_classes: int, weight_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load and initialize the BiSeNet model.

    Args:
        model_name: Name of the backbone model (e.g., "resnet18")
        num_classes: Number of segmentation classes
        weight_path: Path to the model weights file
        device: Device to load the model onto

    Returns:
        torch.nn.Module: Initialized and loaded model
    """
    model = BiSeNet(num_classes, backbone_name=model_name)
    model.to(device)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        raise ValueError(f"Weights not found from given path ({weight_path})")

    model.eval()
    return model


def get_files_to_process(input_path: str) -> List[str]:
    """
    Get a list of image files to process based on the input path.

    Args:
        input_path: Path to a single image file or directory of images

    Returns:
        List[str]: List of file paths to process
    """
    if os.path.isfile(input_path):
        return [input_path]

    # Get all files from the directory
    files = [os.path.join(input_path, f) for f in os.listdir(input_path)]

    # Filter for image files only
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return [f for f in files if os.path.isfile(f) and f.lower().endswith(image_extensions)]


@torch.no_grad()
def inference(params: argparse.Namespace) -> None:
    """
    Run inference on images using the face parsing model.

    Args:
        params: Configuration namespace containing required parameters
    """
    output_path = os.path.join(params.output, params.model)
    os.makedirs(output_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    num_classes = 19  # Number of face parsing classes

    # Load the model
    try:
        model = load_model(params.model, num_classes, params.weight, device)
        logger.info(f"Model loaded successfully: {params.model}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Get list of files to process
    files_to_process = get_files_to_process(params.input)
    logger.info(f"Found {len(files_to_process)} files to process")

    # Process each file
    for file_path in tqdm(files_to_process, desc="Processing images"):
        filename = os.path.basename(file_path)
        save_path = os.path.join(output_path, filename)

        try:
            # Load and process the image
            image = Image.open(file_path).convert("RGB")

            # Store original image resolution
            original_size = image.size  # (width, height)

            # Prepare image for inference
            image_batch = prepare_image(image).to(device)

            # Run inference
            output = model(image_batch)[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only
            predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

            # Convert mask to PIL Image for resizing
            mask_pil = Image.fromarray(predicted_mask.astype(np.uint8))

            # Resize mask back to original image resolution
            restored_mask = mask_pil.resize(original_size, resample=Image.NEAREST)

            # Convert back to numpy array
            predicted_mask = np.array(restored_mask)

            # Visualize and save the results
            vis_parsing_maps(
                image,
                predicted_mask,
                save_image=True,
                save_path=save_path,
            )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    logger.info(f"Processing complete. Results saved to {output_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse and validate command line arguments.

    Returns:
        argparse.Namespace: Validated command line arguments
    """
    parser = argparse.ArgumentParser(description="Face parsing inference")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34"],
        help="model name, i.e resnet18, resnet34"
    )
    parser.add_argument(
        "--weight",
        type=str,
        default="./weights/resnet18.pt",
        help="path to trained model, i.e resnet18/34"
    )
    parser.add_argument("--input", type=str, default="./assets/images/", help="path to an image or a folder of images")
    parser.add_argument("--output", type=str, default="./assets/results", help="path to save model outputs")

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.input):
        raise ValueError(f"Input path does not exist: {args.input}")

    if not os.path.exists(os.path.dirname(args.weight)):
        logger.warning(f"Weight directory does not exist: {os.path.dirname(args.weight)}")

    return args


def main() -> None:
    """Main entry point of the script."""
    try:
        args = parse_args()
        inference(params=args)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
