# Tech Design: BS-RoFormer on Replicate

**Status:** Ready to build
**Visibility:** Public
**Author:** Ryan / FretWise

---

## Goal

Deploy BS-RoFormer as a public Replicate model for SOTA vocals/instrumental separation. Uses the `audio-separator` package for inference on an **A40 Large** GPU. Returns WAV stems only.

---

## Architecture

```
Client (FretWise / Public API)
    │
    ▼
Replicate API  ──webhook──▶  Your webhook endpoint
    │
    ▼
Cog Container (A40 Large GPU)
    ├── audio-separator loads BS-RoFormer
    ├── Separates into vocals + instrumental
    └── Returns WAV stems
```

---

## Model

**`model_bs_roformer_ep_317_sdr_12.9755.ckpt`** — viperx BS-RoFormer 1297, the community consensus best single model for vocals/instrumental separation.

| Model | Vocal SDR | Instrumental SDR |
|---|---|---|
| **BS-Roformer-Viperx-1297** | **12.97** | **17.0** |
| MDX23C-InstVoc HQ 2 | 12.2 | 16.3 |
| htdemucs_ft (Demucs v4) | 10.8 | — |

---

## Repository Structure

```
bs-roformer-replicate/
├── cog.yaml          # Build config — installs audio-separator[gpu], pre-downloads model
├── predict.py        # Cog prediction interface (~35 lines)
├── weights/          # Baked into Docker image at /src/weights/
├── DESIGN.md
└── README.md
```

---

## How It Works

1. **Build time:** `cog.yaml` installs `audio-separator[gpu]` and pre-downloads the BS-RoFormer checkpoint + config to `/src/weights/` using `audio-separator --download_model_only`.

2. **Setup (cold start):** `predict.py` creates a `Separator` instance pointing at `/src/weights/`, loads the model onto GPU.

3. **Predict:** Takes an audio file, calls `separator.separate()`, returns the output WAV files (vocals + instrumental).

---

## Hardware & Cost

| Config | GPU | Cold Start | Inference (5 min song) | Cost/prediction |
|---|---|---|---|---|
| **Recommended** | A40 Large (48GB) | ~30s | ~15-25s | ~$0.02-0.04 |

---

## Deploy Steps

```bash
# 1. Build and test locally
cog build
cog predict -i audio=@test_song.wav

# 2. Push to Replicate
cog login
cog push r8.im/ryansmith/bs-roformer

# 3. Create deployment
# Replicate dashboard → Deployments → New
# Hardware: Nvidia A40 Large
# Min instances: 1 (keeps GPU warm)
# Max instances: 3 (for burst traffic)
```

---

## Calling from FretWise

### Direct call (sync)

```typescript
import Replicate from "replicate";
const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

const output = await replicate.run("ryansmith/bs-roformer", {
  input: { audio: uploadedAudioUrl },
});

// output = [vocals_url, instrumental_url]
```

### Webhook call (async, long audio)

```typescript
const prediction = await replicate.predictions.create({
  model: "ryansmith/bs-roformer",
  input: { audio: uploadedAudioUrl },
  webhook: "https://api.fretwise.ai/webhooks/replicate",
  webhook_events_filter: ["completed"],
});
```
