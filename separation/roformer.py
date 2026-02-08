import torch
import yaml
from bs_roformer import BSRoformer


def load_bs_roformer(config_path: str, checkpoint_path: str, device: torch.device):
    """Load BS-RoFormer from config + checkpoint."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    model = BSRoformer(
        dim=model_cfg["dim"],
        depth=model_cfg["depth"],
        stereo=model_cfg.get("stereo", True),
        num_stems=model_cfg.get("num_stems", 1),
        time_transformer_depth=model_cfg.get("time_transformer_depth", 1),
        freq_transformer_depth=model_cfg.get("freq_transformer_depth", 1),
        dim_head=model_cfg.get("dim_head", 64),
        heads=model_cfg.get("heads", 8),
        ff_dropout=model_cfg.get("ff_dropout", 0.1),
        attn_dropout=model_cfg.get("attn_dropout", 0.1),
        flash_attn=model_cfg.get("flash_attn", True),
        mask_estimator_depth=model_cfg.get("mask_estimator_depth", 2),
    )

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def separate_vocals(
    model, waveform: torch.Tensor, device: torch.device,
    chunk_samples: int = 352800, overlap_samples: int = 176400,
) -> torch.Tensor:
    """
    Run chunked BS-RoFormer inference with overlap-add.

    Args:
        model: Loaded BSRoformer model
        waveform: [2, num_samples] stereo audio at 44100Hz
        device: torch device
        chunk_samples: Chunk length in samples
        overlap_samples: Overlap between chunks in samples

    Returns:
        vocals: [2, num_samples] separated vocal tensor
    """
    channels, total_samples = waveform.shape
    step = chunk_samples - overlap_samples
    vocals = torch.zeros_like(waveform)
    weight = torch.zeros(1, total_samples)

    # Create crossfade window
    window = torch.ones(chunk_samples)
    fade_len = overlap_samples
    fade_in = torch.linspace(0, 1, fade_len)
    fade_out = torch.linspace(1, 0, fade_len)
    window[:fade_len] = fade_in
    window[-fade_len:] = fade_out

    offset = 0
    while offset < total_samples:
        end = min(offset + chunk_samples, total_samples)
        chunk = waveform[:, offset:end]

        # Pad if needed
        if chunk.shape[1] < chunk_samples:
            pad = torch.zeros(channels, chunk_samples - chunk.shape[1])
            chunk = torch.cat([chunk, pad], dim=1)

        with torch.no_grad(), torch.cuda.amp.autocast():
            chunk_out = model(chunk.unsqueeze(0).to(device))
            chunk_out = chunk_out.squeeze(0).cpu()

        # If model returns multiple stems, take first (vocals)
        if chunk_out.dim() == 3:
            chunk_out = chunk_out[0]

        actual_len = end - offset
        w = window[:actual_len]
        vocals[:, offset:end] += chunk_out[:, :actual_len] * w
        weight[:, offset:end] += w

        offset += step

    # Normalize by accumulated weights
    vocals = vocals / weight.clamp(min=1e-8)
    return vocals
