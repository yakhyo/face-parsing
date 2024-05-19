import onnxruntime as ort

import os
import cv2
import argparse
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from utils.common import ATTRIBUTES, COLOR_LIST, letterbox, vis_parsing_maps

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

        resized_image = image.resize((512, 512), resample=Image.BILINEAR)
        transformed_image = prepare_image(resized_image)

        output = onnx_inference(onnx_session, transformed_image)
        predicted_mask = output.squeeze(0).argmax(0)

        vis_parsing_maps(
            resized_image,
            predicted_mask,
            save_image=True,
            save_path=os.path.join(output_path, filename),
        )


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
