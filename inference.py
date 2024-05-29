import os
import argparse
from PIL import Image

import torch
import torchvision.transforms as transforms

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

        resized_image = image.resize((512, 512), resample=Image.BILINEAR)
        transformed_image = prepare_image(resized_image)
        image_batch = transformed_image.to(device)

        output = model(image_batch)[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only
        predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

        vis_parsing_maps(
            resized_image,
            predicted_mask,
            save_image=True,
            save_path=os.path.join(output_path, filename),
        )


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
