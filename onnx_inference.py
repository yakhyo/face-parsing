import os
import argparse
import logging
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort

import torch
import torchvision.transforms as transforms

from utils.common import vis_parsing_maps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def prepare_image(image: Image.Image, input_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Prepare an image for ONNX inference by resizing and normalizing it.
    
    Args:
        image: PIL Image to process
        input_size: Target size for resizing
        
    Returns:
        np.ndarray: Preprocessed image as numpy array ready for model input
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

    return image_batch.numpy()


def load_onnx_model(onnx_path: str) -> ort.InferenceSession:
    """
    Load and initialize the ONNX model.
    
    Args:
        onnx_path: Path to the ONNX model file
        
    Returns:
        ort.InferenceSession: Initialized ONNX inference session
    """
    if not os.path.exists(onnx_path):
        raise ValueError(f"ONNX model not found at path: {onnx_path}")
    
    # Create inference session - use GPU if available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    logger.info(f"ONNX model loaded successfully from {onnx_path}")
    logger.info(f"Using execution provider: {session.get_providers()[0]}")
    
    return session


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


def inference_onnx(params: argparse.Namespace) -> None:
    """
    Run ONNX inference on images using the face parsing model.
    
    Args:
        params: Configuration namespace containing required parameters
    """
    output_path = os.path.join(params.output, os.path.basename(params.model).split('.')[0])
    os.makedirs(output_path, exist_ok=True)
    
    # Load the ONNX model
    try:
        session = load_onnx_model(params.model)
        # Get model input name
        input_name = session.get_inputs()[0].name
        logger.info(f"Model input name: {input_name}")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
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
            image_batch = prepare_image(image)
            
            # Run ONNX inference
            outputs = session.run(None, {input_name: image_batch})
            
            # Get the first output (assuming it's the segmentation map)
            output = outputs[0]
            
            # Convert to segmentation mask
            predicted_mask = output.squeeze(0).argmax(0)
            
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
    parser = argparse.ArgumentParser(description="Face parsing inference with ONNX")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="path to ONNX model file"
    )
    parser.add_argument("--input", type=str, default="./assets/images/", help="path to an image or a folder of images")
    parser.add_argument("--output", type=str, default="./assets/results", help="path to save model outputs")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input):
        raise ValueError(f"Input path does not exist: {args.input}")
    
    if not os.path.exists(args.model):
        raise ValueError(f"ONNX model file does not exist: {args.model}")
    
    return args


def main() -> None:
    """Main entry point of the script."""
    try:
        args = parse_args()
        inference_onnx(params=args)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()