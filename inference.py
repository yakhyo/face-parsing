import os
import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from models.bisenet import BiSeNet

attributes = [
    'skin',
    'l_brow',
    'r_brow',
    'l_eye',
    'r_eye',
    'eye_g',
    'l_ear',
    'r_ear',
    'ear_r',
    'nose',
    'mouth',
    'u_lip',
    'l_lip',
    'neck',
    'neck_l',
    'cloth',
    'hair',
    'hat'
]

COLOR_LIST = [
    [0, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
]


def vis_parsing_maps(image, segmentation_mask, save_image=False, save_path="result.png"):
    # Create numpy arrays for image and segmentation mask
    image = np.array(image).copy().astype(np.uint8)
    segmentation_mask = segmentation_mask.copy().astype(np.uint8)

    # Create a color mask
    segmentation_mask_color = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3))

    num_classes = np.max(segmentation_mask)

    for class_index in range(1, num_classes + 1):
        class_pixels = np.where(segmentation_mask == class_index)
        segmentation_mask_color[class_pixels[0], class_pixels[1], :] = COLOR_LIST[class_index]

    segmentation_mask_color = segmentation_mask_color.astype(np.uint8)

    # Convert image to BGR format for blending
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Blend the image with the segmentation mask
    blended_image = cv2.addWeighted(bgr_image, 0.6, segmentation_mask_color, 0.4, 0)

    # Save the result if required
    if save_image:
        cv2.imwrite(save_path, segmentation_mask)
        cv2.imwrite(save_path, blended_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return blended_image


def letterbox(image, target_size, fill_color=(0, 0, 0)):
    w, h = image.size

    # calculate scale factor based on target aspect ratio
    scale = min(target_size[0] / w, target_size[1] / h)

    # new image dimensions based on scale
    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize the image with antialiasing for better quality
    resized_image = image.resize((new_w, new_h), resample=Image.BILINEAR)

    # calculate padding dimensions
    pad_w = target_size[0] - new_w
    pad_h = target_size[1] - new_h

    # create a new image with target size and fill color
    letterbox_image = Image.new("RGB", target_size, fill_color)

    # paste the resized image at the center of the letterbox image
    letterbox_image.paste(resized_image, (pad_w // 2, pad_h // 2))

    return letterbox_image


def prepare_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image_tensor = transform(image)
    image_batch = image_tensor.unsqueeze(0)

    return image_batch


@torch.no_grad()
def inference(model_path="model_final_diss.pth", input_path="./data", output_path="./results"):
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19

    model = BiSeNet(num_classes, backbone_name='resnet18')
    model.to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path)['model'])
    else:
        raise ValueError(f"Weights not found from given path ({model_path})")

    if os.path.isfile(input_path):
        input_path = [input_path]

    model.eval()
    for filename in os.listdir(input_path):
        image = Image.open(os.path.join(input_path, filename)).convert("RGB")

        resized_image = image.resize((512, 512), resample=Image.BILINEAR)
        transformed_image = prepare_image(resized_image)
        image_batch = transformed_image.to(device)

        output = model(image_batch)[0]  # zero batch
        predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

        vis_parsing_maps(
            resized_image,
            predicted_mask,
            save_image=True,
            save_path=os.path.join(output_path, filename),
        )


if __name__ == "__main__":
    inference(model_path="./weights/checkpoint_resnet18.pth", input_path="images/")
