import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from utils.transform import DefaultTransform


class CelebAMaskHQ(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        if transform is None:
            transform = DefaultTransform()

        self.transform = transform

        self.image_files = []
        self.label_files = []
        for filename in [x for x in os.listdir(self.images_dir) if os.path.splitext(x)[1] in ('.jpg', '.jpeg', '.png')]:
            image_path = os.path.join(self.images_dir, filename)
            label_path = os.path.join(self.labels_dir, f"{filename[:-4]}.png")

            if os.path.isfile(image_path) and os.path.isfile(label_path):
                self.image_files.append(image_path)
                self.label_files.append(label_path)
            else:
                # continue if there is missing image or mask
                continue
                # raise Exception(f"Missing file in labels or images dir: {filename[:-4]}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):

        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        image = Image.open(image_path)
        # mask image size is 512x512, so original image needs to be resized from 1024x1024 to 512x512
        image = image.resize((512, 512), Image.BILINEAR)

        label = Image.open(label_path).convert('P')

        image, label = self.transform(image, label)
        label = np.array(label).astype(np.int64)

        return image, label
