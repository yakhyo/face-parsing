#!/bin/bash

# Check if the weights directory exists, create it if it doesn't
if [ ! -d "weights" ]; then
  mkdir -p weights
fi

# Download ResNet18 weights
if [ ! -f "weights/resnet18.pt" ]; then
  wget -O weights/resnet18.pt https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet18.pt
fi

if [ ! -f "weights/resnet18.onnx" ]; then
  wget -O weights/resnet18.onnx https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet18.onnx
fi

# Download ResNet34 weights
if [ ! -f "weights/resnet34.pt" ]; then
  wget -O weights/resnet34.pt https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet34.pt
fi

if [ ! -f "weights/resnet34.onnx" ]; then
  wget -O weights/resnet34.onnx https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet34.onnx
fi

echo "All weights have been downloaded to the 'weights' folder."
