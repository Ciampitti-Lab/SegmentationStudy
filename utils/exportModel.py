"""Export a PyTorch .pt checkpoint or scripted model to ONNX.

Usage examples:
  python utils/exportModel.py --checkpoint models/corn_SegFormer_seg.pt

If the checkpoint is a state_dict, provide the model builder module and class:
  python utils/exportModel.py \
    --checkpoint models/my_model_epoch40.pt \
    --model-module models.segformer_builder \
    --model-class SegFormer \
    --model-kwargs '{"num_classes":2}' \
    --input-shape 1,3,512,512

If the checkpoint is a scripted/traced model (torch.jit), the exporter will
load it with `torch.jit.load` automatically.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from typing import Tuple, Dict

import torch


def parse_shape(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def load_model_from_checkpoint(checkpoint_path: str, module: str | None, class_name: str | None, kwargs: Dict, device: torch.device):
    # If module+class provided, build model and load state_dict
    if module and class_name:
        mdl = importlib.import_module(module)
        ModelClass = getattr(mdl, class_name)
        model = ModelClass(**kwargs)
        map_location = device
        ck = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(ck, dict) and "state_dict" in ck:
            state = ck["state_dict"]
        else:
            # assume checkpoint is a state_dict
            state = ck
        # Handle possible 'module.' prefix from DataParallel
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            new_state[nk] = v
        model.load_state_dict(new_state)
        return model

    # Try scripted/traced model first
    try:
        model = torch.jit.load(checkpoint_path, map_location=device)
        return model
    except Exception:
        pass

    # Try loading a full model object
    try:
        ck = torch.load(checkpoint_path, map_location=device)
        if isinstance(ck, torch.nn.Module):
            return ck
    except Exception:
        pass

    raise RuntimeError("Unable to load model. Provide --model-module and --model-class when checkpoint only contains state_dict.")


def main():
    p = argparse.ArgumentParser(description="Export PyTorch model (.pt) to ONNX")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint or scripted model")
    p.add_argument("--output", help="Output ONNX path (defaults to same name .onnx)")
    p.add_argument("--model-module", help="Python module path for model builder (e.g. models.segformer)")
    p.add_argument("--model-class", help="Class name inside module to instantiate")
    p.add_argument("--model-kwargs", default="{}", help="JSON dict of kwargs to pass to model class")
    p.add_argument("--input-shape", default="1,3,512,512", help="Comma separated input shape, e.g. 1,3,512,512")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to load model on")
    p.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    p.add_argument("--dynamic", action="store_true", help="Enable dynamic batch dimension for input/output")
    p.add_argument("--input-names", default="input", help="Comma-separated input names")
    p.add_argument("--output-names", default="output", help="Comma-separated output names")
    p.add_argument("--half", action="store_true", help="Convert model and input to float16 before export")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    ckpt = args.checkpoint
    out = args.output or os.path.splitext(ckpt)[0] + ".onnx"
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model_kwargs = json.loads(args.model_kwargs)

    input_shape = parse_shape(args.input_shape)

    print(f"Loading model from: {ckpt}")
    model = load_model_from_checkpoint(ckpt, args.model_module, args.model_class, model_kwargs, device)

    model.to(device)
    model.eval()

    input_tensor = torch.randn(*input_shape, device=device)

    if args.half:
        model.half()
        input_tensor = input_tensor.half()

    input_names = [n.strip() for n in args.input_names.split(",") if n.strip()]
    output_names = [n.strip() for n in args.output_names.split(",") if n.strip()]

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {}
        # set batch dim dynamic for inputs and outputs
        for name in input_names:
            dynamic_axes[name] = {0: "batch_size"}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

    print(f"Exporting to: {out}")
    torch.onnx.export(
        model,
        input_tensor,
        out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=args.verbose,
    )

    print("Export complete.")


if __name__ == "__main__":
    main()
