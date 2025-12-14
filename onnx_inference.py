# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import argparse
import logging
import os
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

from utils.common import vis_parsing_maps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class FaceParsingONNX:
    """Face parsing inference using ONNXRuntime."""

    def __init__(self, model_path: str, session: ort.InferenceSession = None) -> None:
        """Initializes the FaceParsingONNX class.

        Args:
            model_path (str): Path to the ONNX model file.
            session (ort.InferenceSession, optional): ONNX Session. Defaults to None.

        Raises:
            AssertionError: If model_path is None and session is not provided.
            FileNotFoundError: If model_path does not exist.
        """
        self.session = session
        if self.session is None:
            assert model_path is not None, 'Model path is required for the first time initialization.'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'ONNX model not found at path: {model_path}')

            providers = (
                ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if ort.get_device() == 'GPU'
                else ['CPUExecutionProvider']
            )
            self.session = ort.InferenceSession(model_path, providers=providers)

            logger.info(f'ONNX model loaded successfully from {model_path}')
            logger.info(f'Using execution provider: {self.session.get_providers()[0]}')

        self.input_size = (512, 512)
        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name

        outputs = self.session.get_outputs()
        self.output_names = [output.name for output in outputs]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference.

        Args:
            image (np.ndarray): Input image in BGR format (H, W, C).

        Returns:
            np.ndarray: Preprocessed image tensor (1, C, H, W).
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)

        image = image.astype(np.float32) / 255.0
        image = (image - self.input_mean) / self.input_std

        image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        image_batch = np.expand_dims(image, axis=0).astype(np.float32)  # CHW → BCHW

        return image_batch

    def postprocess(self, output: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess model output to segmentation mask.

        Args:
            output (np.ndarray): Raw model output.
            original_size (Tuple[int, int]): Original image size (width, height).

        Returns:
            np.ndarray: Segmentation mask resized to original dimensions.
        """
        predicted_mask = output.squeeze(0).argmax(0).astype(np.uint8)
        restored_mask = cv2.resize(predicted_mask, original_size, interpolation=cv2.INTER_NEAREST)

        return restored_mask

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run face parsing inference on an image.

        Args:
            image (np.ndarray): Input image in BGR format (H, W, C).

        Returns:
            np.ndarray: Segmentation mask with the same size as input image.
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        input_tensor = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        return self.postprocess(outputs[0], original_size)


def get_image_files(input_path: str) -> list:
    """Get a list of image files to process.

    Args:
        input_path (str): Path to a single image file or directory.

    Returns:
        list: List of file paths to process.
    """
    if os.path.isfile(input_path):
        return [input_path]

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = [os.path.join(input_path, f) for f in os.listdir(input_path)]

    return [f for f in files if os.path.isfile(f) and f.lower().endswith(image_extensions)]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Face Parsing ONNX Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model file')
    parser.add_argument('--input', type=str, default='./assets/images/', help='Path to image or folder of images')
    parser.add_argument('--output', type=str, default='./assets/results', help='Path to save results')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise ValueError(f'Input path does not exist: {args.input}')

    if not os.path.exists(args.model):
        raise ValueError(f'ONNX model file does not exist: {args.model}')

    return args


if __name__ == '__main__':
    args = parse_args()

    output_path = os.path.join(args.output, os.path.basename(args.model).split('.')[0])
    os.makedirs(output_path, exist_ok=True)

    engine = FaceParsingONNX(model_path=args.model)

    files_to_process = get_image_files(args.input)
    logger.info(f'Found {len(files_to_process)} files to process')

    for file_path in tqdm(files_to_process, desc='Processing images'):
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)

        save_mask_path = os.path.join(output_path, f'{name}_mask.png')
        save_path = os.path.join(output_path, filename)

        try:
            image = cv2.imread(file_path)
            if image is None:
                logger.warning(f'Failed to read image: {file_path}')
                continue

            mask = engine.predict(image)

            # Save raw mask
            cv2.imwrite(save_mask_path, mask)

            # Visualize and save results
            image_pil = Image.open(file_path).convert('RGB')
            vis_parsing_maps(image_pil, mask, save_image=True, save_path=save_path)

        except Exception as e:
            logger.error(f'Error processing {file_path}: {e}')
            continue

    logger.info(f'Processing complete. Results saved to {output_path}')
