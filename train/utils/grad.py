import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from grepp.model.model import ConvNet, End2End


class GradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        for name, layer in self.target_layers.items():
            def forward_hook(module, input, output, name=name):
                self.activations[name] = output.detach()

            def backward_hook(module, grad_in, grad_out, name=name):
                self.gradients[name] = grad_out[0].detach()

            layer.register_forward_hook(forward_hook)
            layer.register_full_backward_hook(backward_hook)
        
    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        grad_cam_maps = {}
        for name in self.target_layers.keys():
            act = self.activations[name]
            grad = self.gradients[name]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            grad_cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
            grad_cam = torch.nn.functional.interpolate(grad_cam,
                                                       size=input_tensor.shape[2:],
                                                       mode='bilinear',
                                                       align_corners=False)
            grad_cam = grad_cam.squeeze().cpu().numpy()
            grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
            grad_cam_maps[name] = grad_cam
        
        return grad_cam_maps

def get_overlay(cam, raw_image):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + raw_image * 0.6
    padded_overlay = np.pad(np.uint8(overlay),
                            pad_width=((4, 4), (4, 4), (0, 0)),
                            mode='constant',
                            constant_values=tuple((c,) for c in (255,255,255)))
    return padded_overlay


def main():
    base_model = ConvNet()
    model = End2End(base_model, 3)

    print(model)

    image = Image.open('/workspace/grepp/data/val/apple/201_100.jpg')
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),

    ])
    inp_image = transform(image).unsqueeze(0)
    raw_image = np.array(image.resize((64, 64)))

    conv_layers = {
        "conv1": model.base.conv1,
        "conv2": model.base.conv2,
        "conv3": model.base.conv3,
    }

    cam = GradCAM(model, conv_layers)
    cam_maps = cam.generate(inp_image)

    overlays = []
    for name, _ in conv_layers.items():
        cam_map = cam_maps[name]
        overlay = get_overlay(cam_map, raw_image)
        overlays.append(overlay)
        combined_images = np.concatenate(overlays, axis=1)

    cv2.imwrite(f"/workspace/grepp/outputs/gradcam_.jpg",  combined_images)


if __name__=='__main__':
    main()