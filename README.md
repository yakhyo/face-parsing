# BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation [Face Parsing]

![Downloads](https://img.shields.io/github/downloads/yakhyo/face-parsing/total)
[![GitHub Repo stars](https://img.shields.io/github/stars/yakhyo/face-parsing)](https://github.com/yakhyo/face-parsing/stargazers)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/face-parsing)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-yakhyo%2Fface--parsing-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/yakhyo/face-parsing)



<!--
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest updates.</h5>
-->

This is a face parsing model for high-precision facial feature segmentation based on [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897). This model accurately segments various facial components such as the eyes, nose, mouth, and the contour of the face from images. This repo provides a different training & inference code and new backbone model has been added.

<div align="center">
  <img src="assets/slideshow.gif">
</div>

<table>
  <tr>
    <td style="text-align: left;"><p>Input Images</p></td>
    <td style="text-align: center;"><img src="./assets/images/1.jpg" width="200"></td>
    <td style="text-align: center;"><img src="./assets/images/1112.jpg" width="200"></td>
    <td style="text-align: center;"><img src="./assets/images/1309.jpg" width="200"></td>
    <td style="text-align: center;"><img src="./assets/images/1321.jpg" width="200"></td>
  </tr>
  <tr>
    <td style="text-align: left;"><p>ResNet34</p></td>
    <td style="text-align: center;"><img src="./assets/results/resnet34/1.jpg" width="200"></td>
    <td style="text-align: center;"><img src="./assets/results/resnet34/1112.jpg" width="200"></td>
    <td style="text-align: center;"><img src="./assets/results/resnet34/1309.jpg" width="200"></td>
    <td style="text-align: center;"><img src="./assets/results/resnet34/1321.jpg" width="200"></td>
  </tr>
  <tr>
    <td style="text-align: left;"><p>ResNet18</p></td>
    <td style="text-align: center;"><img src="./assets/results/resnet18/1.jpg" width="200"></td>
    <td style="text-align: center;"><img src="./assets/results/resnet18/1112.jpg" width="200"></td>
    <td style="text-align: center;"><img src="./assets/results/resnet18/1309.jpg" width="200"></td>
    <td style="text-align: center;"><img src="./assets/results/resnet18/1321.jpg" width="200"></td>
  </tr>
</table>

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Usage](#usage)
  - [Training](#training)
  - [PyTorch Inference](#pytorch-inference)
  - [ONNX Export](#onnx-export)
  - [ONNX Inference](#onnx-inference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Description

Face parsing model segments facial features with remarkable accuracy, making it ideal for applications in digital
makeup, augmented reality, facial recognition, and emotion detection. The model processes input images and outputs a
detailed mask that highlights individual facial components, distinguishing between skin, hair, eyes, and other key
facial landmarks.

### Recent Updates:

- [2025-03-20] Improved inference code for better performance and efficiency.

### Updates So Far:

- [x] Prepared more clear training code
- [x] Updated backbone models, added ResNet34 model (initially it had only ResNet18)
- [x] Trained model weights/checkpoints with different backbones on [GitHub Release](https://github.com/yakhyo/face-parsing)
- [x] Made several auxiliary updates to the code.
- [x] Torch to ONNX conversion
- [x] ONNX inference

## Installation

To get started with the Face Parsing Model, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yakhyo/face-parsing.git
cd face-parsing
pip install -r requirements.txt
```

## Dataset

This model is designed to work with face parsing datasets. The expected dataset structure should be:

```
dataset/
├── images/           # Input face images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/           # Corresponding segmentation masks
    ├── image1.png
    ├── image2.png
    └── ...
```

This model was trained on:
- **CelebAMask-HQ**: High-quality face parsing dataset with 30,000 images

## Model Performance

| Model    | Parameters | Model Size |
|----------|------------|-----------|
| ResNet18 | ~11.2M     | ~43MB     |
| ResNet34 | ~21.3M     | ~82MB     |

## Usage

#### Download weights (click to download):

| Model    | PT                                                                                         | ONNX                                                                                           |
| -------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| ResNet18 | [resnet18.pt](https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.pt) | [resnet18.onnx](https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx) |
| ResNet34 | [resnet34.pt](https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.pt) | [resnet34.onnx](https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.onnx) |

### Training

Before training, make sure you have prepared your dataset according to the [Dataset](#dataset) section.

Training Arguments:

```
usage: train.py [-h] [--num-classes NUM_CLASSES] [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS] [--image-size IMAGE_SIZE IMAGE_SIZE] [--data-root DATA_ROOT] [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY] [--lr-start LR_START]
                [--max-iter MAX_ITER] [--power POWER] [--lr-warmup-epochs LR_WARMUP_EPOCHS] [--warmup-start-lr WARMUP_START_LR] [--score-thres SCORE_THRES] [--epochs EPOCHS] [--backbone BACKBONE] [--print-freq PRINT_FREQ] [--resume]

Argument Parser for Training Configuration

options:
  -h, --help            show this help message and exit
  --num-classes NUM_CLASSES
                        Number of classes in the dataset (default: 19)
  --batch-size BATCH_SIZE
                        Batch size for training (default: 16)
  --num-workers NUM_WORKERS
                        Number of workers for data loading (default: 4)
  --image-size IMAGE_SIZE IMAGE_SIZE
                        Size of input images (default: 512 512)
  --data-root DATA_ROOT
                        Root directory of the dataset
  --momentum MOMENTUM   Momentum for optimizer (default: 0.9)
  --weight-decay WEIGHT_DECAY
                        Weight decay for optimizer (default: 5e-4)
  --lr-start LR_START   Initial learning rate (default: 1e-2)
  --max-iter MAX_ITER   Maximum number of iterations (default: 80000)
  --power POWER         Power for learning rate policy (default: 0.9)
  --lr-warmup-epochs LR_WARMUP_EPOCHS
                        Number of warmup epochs (default: 10)
  --warmup-start-lr WARMUP_START_LR
                        Warmup starting learning rate (default: 1e-5)
  --score-thres SCORE_THRES
                        Score threshold (default: 0.7)
  --epochs EPOCHS       Number of epochs for training (default: 150)
  --backbone BACKBONE   Backbone architecture (default: resnet18)
  --print-freq PRINT_FREQ
                        Print frequency during training (default: 10)
  --resume              Resume training from checkpoint

```

Start training with default parameters:

```bash
python train.py --data-root /path/to/your/dataset
```

Custom training example:

```bash
python train.py --data-root /path/to/dataset --backbone resnet34 --batch-size 8 --epochs 200
```

### PyTorch Inference

PyTorch Inference Arguments:

```
usage: inference.py [-h] [--model MODEL] [--weight WEIGHT] [--input INPUT] [--output OUTPUT]

Face parsing inference

options:
  -h, --help       show this help message and exit
  --model MODEL    model name, i.e resnet18, resnet34
  --weight WEIGHT  path to trained model, i.e resnet18/34
  --input INPUT    path to an image or a folder of images
  --output OUTPUT  path to save model outputs

```

PyTorch inference examples:

Single image:
```bash
python inference.py --model resnet18 --weight ./weights/resnet18.pt --input ./assets/images/1.jpg --output ./results
```

Batch processing:
```bash
python inference.py --model resnet18 --weight ./weights/resnet18.pt --input ./assets/images --output ./assets/results
```

### ONNX Export

Convert PyTorch models to ONNX format for cross-platform deployment:

```
usage: onnx_export.py [-h] [--model MODEL] [--weight WEIGHT]

Convert PyTorch model to ONNX format

options:
  -h, --help       show this help message and exit
  --model MODEL    model name, i.e resnet18, resnet34 (default: resnet18)
  --weight WEIGHT  path to trained PyTorch model (default: ./weights/resnet18.pt)
```

Export examples:

```bash
# Export ResNet18 model
python onnx_export.py --model resnet18 --weight ./weights/resnet18.pt

# Export ResNet34 model
python onnx_export.py --model resnet34 --weight ./weights/resnet34.pt
```

This will create an ONNX file in the same directory as the PyTorch model (e.g., `resnet18.onnx`).

### ONNX Inference

ONNX inference arguments:

```
usage: onnx_inference.py [-h] --model MODEL [--input INPUT] [--output OUTPUT]

Face parsing inference with ONNX

options:
  -h, --help       show this help message and exit
  --model MODEL    path to ONNX model file
  --input INPUT    path to an image or a folder of images
  --output OUTPUT  path to save model outputs
```

ONNX inference example:

```bash
python onnx_inference.py --model ./weights/resnet18.onnx --input ./assets/images --output ./assets/results/resnet18onnx
```

## Project Structure

```
face-parsing/
├── models/                 # Model architecture definitions
│   ├── bisenet.py         # BiSeNet implementation
│   └── resnet.py          # ResNet backbone implementations
├── utils/                  # Utility modules
│   ├── common.py          # Common utility functions
│   ├── dataset.py         # Dataset loading and preprocessing
│   ├── loss.py            # Loss function definitions
│   ├── prepare_labels.py  # Label preparation utilities
│   └── transform.py       # Image transformation functions
├── assets/                 # Demo images and results
│   ├── images/            # Sample input images
│   ├── results/           # Sample output results
│   └── slideshow.gif      # Demo animation
├── weights/                # Model checkpoints (download separately)
├── train.py               # Training script
├── inference.py           # PyTorch inference script
├── onnx_export.py         # PyTorch to ONNX conversion
├── onnx_inference.py      # ONNX inference script
├── download.sh            # Weight download script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Acknowledged By

- The [facefusion/facefusion](https://github.com/facefusion/facefusion) (with over 20k stars) uses the main face-parsing module from this repository.
- The [FermatResearch/BiSeNet-Cog](https://github.com/FermatResearch/BiSeNet-Cog) provides a Cog implementation for containerized deployment of this BiSeNet model.

## Contributing

Contributions to improve the Face Parsing Model are welcome. Feel free to fork the repository and submit pull requests,
or open issues to suggest features or report bugs.

## License

The project is licensed under the [MIT license](https://opensource.org/license/mit/).

## Citation

```
@misc{face-parsing,
  author = {Valikhujaev Yakhyokhuja},
  title = {face-parsing},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yakhyo/face-parsing}},
  note = {GitHub repository}
}

```

## Reference

The project is built on top of [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch). Model architecture and training strategy have been re-written for better performance.

<!--
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yakhyo/face-parsing&type=Date)](https://star-history.com/#yakhyo/face-parsing&Date)
-->

