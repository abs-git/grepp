import argparse
import os

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from grepp.model.model import ConvNet, End2End

class CalibReader(CalibrationDataReader):
    def __init__(self, calibration_inputs):
        self.data = iter([{"input": x} for x in calibration_inputs])

    def get_next(self):
        return next(self.data, None)

def main(args):

    # setting
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    image_dir = args.image_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model_name = checkpoint_path.split('/')[-1].split('.')[0]
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    onnx_int8_path = os.path.join(output_dir, f"{model_name}_int8.onnx")

    input_name = 'input'
    output_name = 'output0'

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # model load
    base_model = ConvNet()
    model = End2End(base_model, num_classes=3).to(device)
    ckpt = torch.load(checkpoint_path,weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    # onnx export
    dummy = torch.randn(1, 3, 64, 64).to(device)

    dynamic = True
    if dynamic:
        dynamic_axes = {input_name: {0: "batch_size"},
                        output_name: {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes if dynamic else None
    )

    # int8 quantization
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    calibration_samples = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)

        image_tensor = transform(image).unsqueeze(0)
        calibration_samples.append(image_tensor.numpy().astype(np.float32))

    reader = CalibReader(calibration_samples)

    quantize_static(
        model_input=onnx_path,
        model_output=onnx_int8_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    main(args)