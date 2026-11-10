"""Strict compatibility adapter for KemeTone's parametrized weight-normalization checkpoint.

Kokoro 0.9.4 uses legacy weight_g/weight_v names. Its fallback loader can silently
skip newer names. Map only the equivalent PyTorch representation, and require
every remaining tensor to match. Non-affine InstanceNorm is represented by the
legacy module's identity scale/zero bias. No learned weight is fabricated.
"""

from pathlib import Path


def load_verified_weights(model, checkpoint_path: Path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_components = {"bert", "bert_encoder", "predictor", "text_encoder", "decoder"}
    if set(checkpoint) != expected_components:
        raise ValueError("Unsupported KemeTone checkpoint components. Install the pinned voice revision.")
    for component, tensors in checkpoint.items():
        module = getattr(model, component)
        expected = module.state_dict()
        mapped = {}
        for name, value in tensors.items():
            name = name.removeprefix("module.")
            name = name.replace(".parametrizations.weight.original0", ".weight_g")
            name = name.replace(".parametrizations.weight.original1", ".weight_v")
            if name in mapped:
                raise ValueError("Duplicate voice checkpoint tensor.")
            mapped[name] = value
        for name in set(expected) - set(mapped):
            parent, field = name.rsplit(".", 1)
            layer = module.get_submodule(parent)
            if isinstance(layer, torch.nn.InstanceNorm1d) and field in ("weight", "bias"):
                mapped[name] = (
                    torch.ones_like(expected[name]) if field == "weight" else torch.zeros_like(expected[name])
                )
            else:
                raise ValueError(f"Voice checkpoint is missing a required tensor: {component}.{name}")
        try:
            module.load_state_dict(mapped, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Voice checkpoint is incompatible with {component}. Reinstall the pinned speech environment."
            ) from exc
    return model
