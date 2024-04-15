# Face Parsing Model

This is a face parsing model for high-precision facial feature segmentation built on top
of [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch). This model accurately segments various
facial components such as the eyes, nose, mouth, and the contour of the face from images. This repo provides a different
training & inference code and new backbone model has been added.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description

Face parsing model segments facial features with remarkable accuracy, making it ideal for applications in digital
makeup, augmented reality, facial recognition, and emotion detection. The model processes input images and outputs a
detailed mask that highlights individual facial components, distinguishing between skin, hair, eyes, and other key
facial landmarks.

Following updates have been made so far:

- [x] Prepared more clear training code
- [x] Updated backbone models, added resnet34 model (initially it has only resnet18)
- [x] Trained model with different backbones are
  given [here](https://github.com/yakhyo/face-parsing/releases/tag/v0.0.1)
- [x] Made several auxiliary updates to the code.

### ToDo
- [] torch to onnx convert
- [] onnx inference

## Installation

To get started with the Face Parsing Model, clone this repository and install the required dependencies:

```commandline
git clone https://github.com/yakhyo/face-parsing.git
cd face-parsing-model
pip install -r requirements.txt
```

## Usage

### Train

Training Arguments:

```
usage: train.py [-h] [--num-classes NUM_CLASSES] [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS] [--image-size IMAGE_SIZE IMAGE_SIZE] [--data-root DATA_ROOT] [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY] [--lr-start LR_START]
                [--max-iter MAX_ITER] [--power POWER] [--lr-warmup-epochs LR_WARMUP_EPOCHS] [--warmup-start-lr WARMUP_START_LR] [--score-thres SCORE_THRES] [--epochs EPOCHS] [--backbone BACKBONE] [--print-freq PRINT_FREQ] [--resume]

Argument Parser for Training Configuration

options:
  -h, --help            show this help message and exit
  --num-classes NUM_CLASSES
                        Number of classes in the dataset
  --batch-size BATCH_SIZE
                        Batch size for training
  --num-workers NUM_WORKERS
                        Number of workers for data loading
  --image-size IMAGE_SIZE IMAGE_SIZE
                        Size of input images
  --data-root DATA_ROOT
                        Root directory of the dataset
  --momentum MOMENTUM   Momentum for optimizer
  --weight-decay WEIGHT_DECAY
                        Weight decay for optimizer
  --lr-start LR_START   Initial learning rate
  --max-iter MAX_ITER   Maximum number of iterations
  --power POWER         Power for learning rate policy
  --lr-warmup-epochs LR_WARMUP_EPOCHS
                        Number of warmup epochs
  --warmup-start-lr WARMUP_START_LR
                        Warmup starting learning rate
  --score-thres SCORE_THRES
                        Score threshold
  --epochs EPOCHS       Number of epochs for training
  --backbone BACKBONE   Backbone architecture
  --print-freq PRINT_FREQ
                        Print frequency during training
  --resume              Resume training from checkpoint

```

```commandline
python train.py
```

### Inference

Inference Arguments:

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

```
python inference.py --model resnet18 --weights ./weights/resnet18.pt --input assets/images --output assets/results
```

<div align='center'>
<p>Model Inference Results</p>
    <img src='./assets/images/1.jpg' height="200px">
    <img src='./assets/images/1112.jpg' height="200px">
    <img src='./assets/images/1309.jpg' height="200px">
    <img src='./assets/images/1321.jpg' height="200px">

<p>Backbone:ResNet34 </p>
    <img src='./assets/results/resnet34/1.jpg' height="200px">
    <img src='./assets/results/resnet34/1112.jpg' height="200px">
    <img src='./assets/results/resnet34/1309.jpg' height="200px">
    <img src='./assets/results/resnet34/1321.jpg' height="200px">

<p>Backbone:ResNet18 </p>
    <img src='./assets/results/resnet18/1.jpg' height="200px">
    <img src='./assets/results/resnet18/1112.jpg' height="200px">
    <img src='./assets/results/resnet18/1309.jpg' height="200px">
    <img src='./assets/results/resnet18/1321.jpg' height="200px">
</div>

## Contributing

Contributions to improve the Face Parsing Model are welcome. Feel free to fork the repository and submit pull requests,
or open issues to suggest features or report bugs.

## License

The project is licensed under the [MIT license](https://opensource.org/license/mit/).
