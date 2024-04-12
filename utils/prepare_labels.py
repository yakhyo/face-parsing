import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing

face_sep_mask = './Datasets/CelebAMask-HQ-mask-anno'  # specify the path
mask_path = './Datasets/CelebAMask-HQ/mask'  # specify the path to save result masks

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


def process_folder(i):
    count, total = 0, 0  # Initialize count and total for each folder
    for j in tqdm(range(i * 2000, (i + 1) * 2000), desc=f"Processing Folder {i}"):
        mask = np.zeros((512, 512))
        for idx, attribute in enumerate(attributes, 1):
            total += 1
            filename = f"{str(j).zfill(5)}_{attribute}.png"
            path = os.path.join(face_sep_mask, str(i), filename)
            if os.path.exists(path):
                count += 1
                sep_mask = np.array(Image.open(path).convert('P'))
                mask[sep_mask == 225] = idx
        cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
    return count, total


def process_folders_with_multiprocessing(folders_to_process=15):
    """Processes multiple folders using multiprocessing."""

    with multiprocessing.Pool() as pool:
        results = pool.map(process_folder, range(folders_to_process))

    # Collect and aggregate results from each folder
    count = 0
    total = 0
    for c, t in results:
        count += c
        total += t

    print(f"Total files processed: {count}, {total}")


def process_folder_default(folders_to_process=15):
    count, total = 0, 0
    os.makedirs(mask_path, exist_ok=True)
    for i in range(folders_to_process):
        c, t = process_folder(i)
        count += c
        total += t
    print(count, total)


if __name__ == "__main__":
    # process_folder_default() # without multiprocessing
    process_folders_with_multiprocessing()  # with multiprocessing
