import random
import numpy as np
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as F

__all__ = ["TrainTransform", "DefaultTransform"]


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        assert image.size == target.size, "Image and target size mismatch"

        crop_width, crop_height = self.size
        width, height = image.size

        if image.size == self.size:
            return image, target

        # Handle cases where the image is smaller than the desired crop size
        if width < crop_width or height < crop_height:
            scale = max(crop_width / width, crop_height / height)
            new_width, new_height = int(scale * width), int(scale * height)

            image = image.resize((new_width, new_height), Image.BILINEAR)
            target = target.resize((new_width, new_height), Image.NEAREST)

            width, height = new_width, new_height

        # Randomly select the top-left corner of the crop region
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)

        # Crop both image and target
        image = image.crop((left, top, left + crop_width, top + crop_height))
        target = target.crop((left, top, left + crop_width, top + crop_height))

        return image, target


class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            np_target = np.array(target)
            label_swaps = {
                2: 3,  # l -> r eyebrow
                3: 2,  # r -> l eyebrow
                4: 5,  # l -> r eye
                5: 4,  # r -> l eye
                7: 8,  # l -> r ear
                8: 7,  # r -> l ear
            }
            np_target_flipped = np_target.copy()

            for src, dst in label_swaps.items():
                np_target_flipped[np_target == src] = dst 

            target = Image.fromarray(np_target_flipped)

            # flip image and image mask
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        return image, target


class RandomScale:
    def __init__(self, scales=(1,)):
        self.scales = scales

    def __call__(self, image, target):
        width, height = image.size
        scale = random.choice(self.scales)

        new_width, new_height = int(width * scale), int(height * scale)

        image = image.resize((new_width, new_height), Image.BILINEAR)
        target = target.resize((new_width, new_height), Image.NEAREST)

        return image, target


class ColorJitter:
    def __init__(self, brightness=None, contrast=None, saturation=None):
        if brightness is not None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if contrast is not None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if saturation is not None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, image, target):

        brightness = random.uniform(self.brightness[0], self.brightness[1])
        contrast = random.uniform(self.contrast[0], self.contrast[1])
        saturation = random.uniform(self.saturation[0], self.saturation[1])

        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)

        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class TrainTransform:
    def __init__(self, image_size):
        self.transform = Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            HorizontalFlip(p=0.5),
            RandomScale([0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
            RandomCrop(image_size),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, image, label):
        return self.transform(image, label)


class DefaultTransform:
    def __init__(self):
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, image, label):
        return self.transform(image, label)
