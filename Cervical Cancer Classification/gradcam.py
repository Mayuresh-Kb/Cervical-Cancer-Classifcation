"""Generate a Grad-CAM explanation for a trained CNN prediction."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from experiment_utils import MEAN, STD, build_model, device_from, make_transforms

def target_layer(model, name):
    if name == "resnet18": return model.layer4[-1]
    if name == "densenet121": return model.features.norm5
    if name == "efficientnet_b0": return model.features[-1]
    if name == "convnext_tiny": return model.features[-1]
    if name == "mobilenet_v3_large": return model.features[-1]
    if name == "vgg16": return model.features[-1]
    raise ValueError("Grad-CAM is implemented for CNN models, not swin_t.")

def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--image", type=Path, required=True); p.add_argument("--output", type=Path, default=Path("gradcam.png")); p.add_argument("--class-index", type=int); p.add_argument("--device")
    args = p.parse_args(); device = device_from(args.device); checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False); classes = checkpoint["classes"]; model = build_model(checkpoint["model"], len(classes), False).to(device); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    activations, gradients = [], []
    layer = target_layer(model, checkpoint["model"]); layer.register_forward_hook(lambda _, __, output: activations.append(output)); layer.register_full_backward_hook(lambda _, __, output: gradients.append(output[0]))
    _, transform = make_transforms(); original = Image.open(args.image).convert("RGB"); tensor = transform(original).unsqueeze(0).to(device); logits = model(tensor); index = args.class_index if args.class_index is not None else int(logits.argmax(1).item()); model.zero_grad(); logits[0, index].backward()
    activation, gradient = activations[-1][0], gradients[-1][0]; weights = gradient.mean(dim=tuple(range(1, gradient.ndim)), keepdim=True); heatmap = torch.relu((weights * activation).sum(0)).detach().cpu().numpy(); heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(original.size, Image.Resampling.BILINEAR); figure, axis = plt.subplots(figsize=(6, 6)); axis.imshow(original); axis.imshow(heatmap_image, cmap="jet", alpha=.45); axis.set_title(f"Grad-CAM: {classes[index]}"); axis.axis("off"); figure.tight_layout(); figure.savefig(args.output, dpi=200); plt.close(figure)
    print(f"Wrote {args.output}")

if __name__ == "__main__": main()
