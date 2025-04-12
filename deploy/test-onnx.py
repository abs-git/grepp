import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image


def main(args):

    onnx_path = args.onnx_path
    image_dir = args.image_dir

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    classes = {
        'tomato': 0,
        'cherry': 1,
        'apple': 2
    }

    results = {}
    for c in os.listdir(image_dir):

        if c not in results:
            results[c] = 0.0

        collect, total = 0, 0

        class_dir = os.path.join(image_dir, c)
        pathes = sorted(os.listdir(class_dir))
        for p in pathes:
            image_path = os.path.join(class_dir, p)

            image = Image.open(image_path)

            image_tensor = transform(image).unsqueeze(0)
            image_np = image_tensor.numpy().astype(np.float32)

            outputs = session.run(None, {input_name: image_np})[0]

            preds = np.argmax(outputs[0], axis=0)

            if classes[c] == preds:
                collect += 1
            total += 1

        results[c] = round((collect / total)*100, 2)

    print("results: ", results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    main(args)