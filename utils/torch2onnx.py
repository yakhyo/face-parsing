import argparse

import torch

from models.bisenet import BiSeNet


def torch2onnx_export(params):
    num_classes = 19

    model = BiSeNet(num_classes, backbone_name=params.model)
    model.load_state_dict(torch.load(params.weight))

    onnx_model_path = params.weight.replace(".pt", ".onnx")

    dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True)

    # Export the model to ONNX
    torch.onnx.export(model,                # model being run
                      dummy_input,          # model input (or a tuple for multiple inputs)
                      onnx_model_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,   # store the trained parameter weights inside the model file
                      opset_version=15,     # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                    'output': {0: 'batch_size'}})


def parse_args():
    parser = argparse.ArgumentParser(description="Face parsing inference")
    parser.add_argument("--model", type=str, default="resnet18", help="model name, i.e resnet18, resnet34")
    parser.add_argument(
        "--weight",
        type=str,
        default="./weights/resnet18.pt",
        help="path to trained model, i.e resnet18/34"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch2onnx_export(params=args)
