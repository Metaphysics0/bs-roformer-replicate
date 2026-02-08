# Tech Design: BS-RoFormer on Replicate

**Status:** Ready to build
**Visibility:** Public
**Author:** Ryan / FretWise

---

## Goal

Deploy BS-RoFormer as the first public Replicate model for SOTA music source separation. Supports multi-stem separation (vocals, drums, bass, guitar, piano, other). Keeps GPU warm alongside existing FretWise Modal infra and provides a public API with webhook support for async processing.

---

## Architecture

```
Client (FretWise / Public API)
    │
    ▼
Replicate API  ──webhook──▶  Your webhook endpoint
    │
    ▼
Cog Container (T4 GPU)
    ├── Load audio (URL or file upload)
    ├── Resample to 44.1kHz stereo
    ├── Run separation model(s) in sequence
    └── Return separated stems as WAV/MP3/FLAC
```

### Webhook Flow (for long audio)

```
POST replicate.com/v1/predictions
  { "input": {...}, "webhook": "https://api.fretwise.ai/webhooks/replicate" }
    │
    ▼
Replicate queues prediction → runs on GPU → POST result to webhook URL
```

---

## Model Selection

### Primary Model: `model_bs_roformer_ep_317_sdr_12.9755.ckpt`

This is the **viperx BS-RoFormer 1297** — the community consensus best single model for vocals/instrumental separation. It leads the SDR charts across every benchmark:

| Model | Vocal SDR | Instrumental SDR |
|---|---|---|
| **BS-Roformer-Viperx-1297** | **12.97** | **17.0** |
| BS-Roformer-Viperx-1296 | 12.96 | 17.0 |
| MDX23C-InstVoc HQ 2 | 12.2 | 16.3 |
| htdemucs_ft (Demucs v4) | 10.8 | — |

### Multi-Stem Strategy

There's no single BS-RoFormer checkpoint that does all stems at once with top quality. The community standard (used by MVSEP, UVR, and every serious separation pipeline) is to **run specialized models per stem**. This is what we'll do:

| Stem | Model | Architecture | Source |
|---|---|---|---|
| **Vocals / Instrumental** | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | BS-RoFormer | viperx |
| **Drums** | `htdemucs_ft.yaml` | Demucs v4 | Meta (drums SDR: 10.1) |
| **Bass** | `htdemucs_ft.yaml` | Demucs v4 | Meta (bass SDR: 11.9) |
| **Guitar + Piano** | `htdemucs_6s.yaml` | Demucs v4 6-stem | Meta |

The pipeline for full multi-stem is:

```
Input audio
    │
    ├──▶ BS-RoFormer → vocals, instrumental
    │
    └──▶ Demucs htdemucs_ft → drums, bass, other
         (or htdemucs_6s → drums, bass, guitar, piano, other)
```

This gives you the best of both worlds: SOTA vocal separation from BS-RoFormer, and solid drum/bass/guitar from Demucs. The client just sends one request and gets back all stems.

---

## Repository Structure

```
bs-roformer-cog/
├── cog.yaml
├── predict.py                 # Cog prediction interface
├── weights/                   # Baked into Docker image
│   └── model_bs_roformer_ep_317_sdr_12.9755.ckpt
├── configs/
│   └── config_bs_roformer_vocals.yaml
├── separation/
│   ├── roformer.py            # BS-RoFormer inference (from ZFTurbo)
│   └── demucs_wrapper.py      # Demucs inference wrapper
└── README.md
```

---

## cog.yaml

```yaml
build:
  gpu: true
  cuda: "12.1"
  python_version: "3.11"
  system_packages:
    - "ffmpeg"
    - "libsndfile1"
  python_packages:
    - "torch==2.1.2"
    - "torchaudio==2.1.2"
    - "numpy<2"
    - "librosa==0.10.1"
    - "soundfile==0.12.1"
    - "pyyaml==6.0.1"
    - "einops>=0.7.0"
    - "bs-roformer==1.1.0"
    - "rotary-embedding-torch"
    - "demucs"
  run:
    # Download and cache the BS-RoFormer weights at build time
    - "mkdir -p /src/weights"
    - "curl -L -o /src/weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt https://huggingface.co/viperx/bs-roformer/resolve/main/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
predict: "predict.py:Predictor"
```

---

## predict.py

```python
import os
import tempfile
import subprocess
from typing import List
from cog import BasePredictor, Input, Path
import torch
import torchaudio
import numpy as np


class Predictor(BasePredictor):

    def setup(self):
        """Load BS-RoFormer into GPU on cold start. Demucs loads lazily."""
        self.device = torch.device("cuda")

        # Load BS-RoFormer
        from separation.roformer import load_bs_roformer
        self.bs_roformer = load_bs_roformer(
            config_path="configs/config_bs_roformer_vocals.yaml",
            checkpoint_path="weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            device=self.device,
        )

        # Pre-load Demucs (htdemucs_ft downloads on first use, ~80MB)
        from demucs.pretrained import get_model
        self.demucs_4stem = get_model("htdemucs_ft")
        self.demucs_4stem.to(self.device).eval()

        self.demucs_6stem = None  # Lazy load only if guitar/piano requested

    def predict(
        self,
        audio: Path = Input(
            description="Audio file to separate (WAV, MP3, FLAC, OGG, etc.)"
        ),
        stems: str = Input(
            description="Which stems to extract",
            default="vocals_instrumental",
            choices=[
                "vocals_instrumental",   # Just vocals + instrumental
                "all_4",                 # vocals, drums, bass, other
                "all_6",                 # vocals, drums, bass, guitar, piano, other
            ],
        ),
        output_format: str = Input(
            description="Output audio format",
            default="wav",
            choices=["wav", "mp3", "flac"],
        ),
        chunk_size: int = Input(
            description="Processing chunk size in seconds. Larger = better quality but more VRAM.",
            default=8,
            ge=4,
            le=30,
        ),
        overlap: int = Input(
            description="Overlap between chunks (higher = fewer artifacts at boundaries)",
            default=4,
            ge=1,
            le=8,
        ),
        sample_rate: int = Input(
            description="Output sample rate",
            default=44100,
            choices=[44100, 48000],
        ),
    ) -> List[Path]:
        """Separate audio into stems."""

        # --- 1. Load & preprocess ---
        waveform, sr = torchaudio.load(str(audio))
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        if waveform.shape[0] > 2:
            waveform = waveform[:2]  # Take first 2 channels

        out_dir = tempfile.mkdtemp()
        outputs = []

        # --- 2. Run BS-RoFormer for vocals ---
        from separation.roformer import separate_vocals
        vocals = separate_vocals(
            self.bs_roformer, waveform, self.device,
            chunk_samples=chunk_size * 44100,
            overlap_samples=overlap * 44100,
        )
        instrumental = waveform - vocals

        if stems == "vocals_instrumental":
            outputs.extend(
                self._save_stems(out_dir, output_format, sample_rate,
                                 vocals=vocals, instrumental=instrumental)
            )
            return outputs

        # --- 3. Run Demucs for drums/bass/other (and optionally guitar/piano) ---
        if stems == "all_6":
            if self.demucs_6stem is None:
                from demucs.pretrained import get_model
                self.demucs_6stem = get_model("htdemucs_6s")
                self.demucs_6stem.to(self.device).eval()
            demucs_model = self.demucs_6stem
            stem_names = ["drums", "bass", "other", "vocals", "guitar", "piano"]
        else:
            demucs_model = self.demucs_4stem
            stem_names = ["drums", "bass", "other", "vocals"]

        from demucs.apply import apply_model
        demucs_input = waveform.unsqueeze(0).to(self.device)
        with torch.no_grad():
            demucs_out = apply_model(
                demucs_model, demucs_input,
                shifts=1, overlap=0.25
            )[0]  # shape: [num_stems, channels, samples]

        # Map Demucs outputs to dict
        demucs_stems = {}
        for i, name in enumerate(stem_names):
            demucs_stems[name] = demucs_out[i].cpu()

        # --- 4. Combine: BS-RoFormer vocals + Demucs everything else ---
        final_stems = {"vocals": vocals}
        for name in ["drums", "bass", "other"]:
            if name in demucs_stems:
                final_stems[name] = demucs_stems[name]
        if stems == "all_6":
            for name in ["guitar", "piano"]:
                if name in demucs_stems:
                    final_stems[name] = demucs_stems[name]

        outputs.extend(
            self._save_stems(out_dir, output_format, sample_rate, **final_stems)
        )
        return outputs

    def _save_stems(self, out_dir, fmt, sr, **stems) -> List[Path]:
        """Save stem tensors to audio files."""
        paths = []
        for name, tensor in stems.items():
            if sr != 44100:
                tensor = torchaudio.functional.resample(tensor, 44100, sr)
            path = os.path.join(out_dir, f"{name}.{fmt}")
            if fmt == "mp3":
                # Save as wav first, then convert with ffmpeg
                wav_path = path.replace(".mp3", ".wav")
                torchaudio.save(wav_path, tensor, sr)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", wav_path, "-b:a", "320k", path],
                    capture_output=True,
                )
                os.remove(wav_path)
            else:
                torchaudio.save(path, tensor, sr)
            paths.append(Path(path))
        return paths
```

---

## separation/roformer.py

Core inference logic, adapted from ZFTurbo's `Music-Source-Separation-Training`:

```python
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
```

---

## Hardware & Cost

| Config | GPU | Cold Start | Inference (5 min song) | Cost/prediction |
|---|---|---|---|---|
| **Recommended** | T4 (16GB) | ~30s | ~45-60s (vocals only), ~90s (all_4) | ~$0.01-0.02 |
| Performance | A40 (48GB) | ~30s | ~15-25s (vocals only), ~45s (all_4) | ~$0.02-0.04 |

BS-RoFormer is ~70M params (~280MB in memory). Demucs htdemucs_ft is ~83M params. Both fit comfortably on a T4 together. The 6-stem Demucs model is loaded lazily only when `all_6` is requested.

**GPU warm strategy:** Replicate deployments let you set a minimum instance count. Set `min_instances=1` to keep one T4 warm at all times (~$0.33/hr = ~$240/mo). Scale to 0 if you want cold start only.

---

## Deploy Steps

```bash
# 1. Clone and setup
git clone <your-repo>
cd bs-roformer-cog

# 2. Install Cog
sudo curl -o /usr/local/bin/cog -L \
  https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog

# 3. Build and test locally
cog build
cog predict -i audio=@test_song.wav -i stems=vocals_instrumental

# 4. Create public model on Replicate
# https://replicate.com/create → "ryansmith/bs-roformer"
# Set visibility: public

# 5. Push
cog login
cog push r8.im/ryansmith/bs-roformer

# 6. Create deployment (for warm GPU)
# Replicate dashboard → Deployments → New
# Hardware: Nvidia T4
# Min instances: 1 (keeps GPU warm)
# Max instances: 3 (for burst traffic)
```

---

## Calling from FretWise

### Direct call (sync, <60s audio)

```typescript
import Replicate from "replicate";
const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

const output = await replicate.run("ryansmith/bs-roformer", {
  input: {
    audio: uploadedAudioUrl,
    stems: "vocals_instrumental",
    output_format: "wav",
  },
});

// output = [vocals_url, instrumental_url]
const instrumentalUrl = output[1];
// Feed to YourMT3+ for guitar transcription
```

### Webhook call (async, long audio)

```typescript
const prediction = await replicate.predictions.create({
  model: "ryansmith/bs-roformer",
  input: {
    audio: uploadedAudioUrl,
    stems: "all_4",
    output_format: "wav",
  },
  webhook: "https://api.fretwise.ai/webhooks/replicate",
  webhook_events_filter: ["completed"],
});

// prediction.id → store in DB, match when webhook fires
```

### Webhook handler (Cloudflare Worker)

```typescript
export default {
  async fetch(request: Request, env: Env) {
    const body = await request.json();

    if (body.status === "succeeded") {
      const { id, output } = body;
      // output = [vocals_url, drums_url, bass_url, other_url]

      // Store stem URLs, trigger next pipeline step
      await env.DB.prepare(
        "UPDATE jobs SET stems = ?, status = 'stems_ready' WHERE replicate_id = ?"
      ).bind(JSON.stringify(output), id).run();

      // Trigger YourMT3+ transcription on Modal
      await triggerTranscription(env, id, output);
    }

    return new Response("ok", { status: 200 });
  },
};
```

---

## FretWise Pipeline Integration

```
User uploads audio
    │
    ▼
Cloudflare Worker receives file
    │
    ▼
POST to Replicate (bs-roformer, stems=all_4, webhook=...)
    │
    ▼
Replicate runs BS-RoFormer + Demucs on T4
    │
    ▼
Webhook fires → stems_ready
    │
    ▼
Feed instrumental stem to YourMT3+ on Modal
    │
    ▼
Convert MIDI → Guitar Pro
    │
    ▼
Return tabs to user
```

This keeps BS-RoFormer on Replicate (warm GPU, public API, community visibility) and YourMT3+ on Modal (your custom fine-tuned models, existing infra).

---

## Timeline

| Task | Effort |
|---|---|
| Repo setup, cog.yaml, download weights | 1 hour |
| Adapt ZFTurbo inference → `separation/roformer.py` | 2 hours |
| Write `predict.py` with multi-stem pipeline | 2 hours |
| Local testing with `cog predict` | 1 hour |
| Push to Replicate, test hosted API | 1 hour |
| Webhook handler in Cloudflare Worker | 1 hour |
| README + public model page | 30 min |
| **Total** | **~1 day** |

---

## Open Questions

1. **Demucs licensing** — Demucs is MIT licensed, BS-RoFormer repo is MIT, viperx weights are open. Should be clear for public hosting but worth double-checking weight licenses.
2. **Cold start optimization** — Baking both BS-RoFormer + Demucs weights into the image makes it ~500MB+. Consider using Replicate's `weights` URL feature to lazy-load from R2/S3 if image size becomes a problem.
3. **Future: replace Demucs stems** — As community trains better BS-RoFormer/MelBand-RoFormer checkpoints for drums, bass, guitar, you can swap out Demucs per-stem without changing the API interface.