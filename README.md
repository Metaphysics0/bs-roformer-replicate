# BS-RoFormer — Music Source Separation

State-of-the-art music source separation using [BS-RoFormer](https://arxiv.org/abs/2309.02612) (Band-Split RoPE Transformer) and [Demucs v4](https://github.com/facebookresearch/demucs). Deployed on [Replicate](https://replicate.com/metaphysics0/bs-roformer) for easy API access.

## What it does

Separates a song into individual stems:

- **`vocals_instrumental`** — Vocals + instrumental (BS-RoFormer only, fastest)
- **`all_4`** — Vocals, drums, bass, other (BS-RoFormer + Demucs htdemucs_ft)
- **`all_6`** — Vocals, drums, bass, guitar, piano, other (BS-RoFormer + Demucs 6-stem)

Vocals are separated by BS-RoFormer (viperx-1297, SDR 12.97 — the best public checkpoint). Drums, bass, and other stems come from Demucs v4 which leads on those instruments.

## Models

| Stem | Model | SDR |
|---|---|---|
| Vocals | BS-RoFormer viperx-1297 | 12.97 |
| Drums | Demucs htdemucs_ft | 10.1 |
| Bass | Demucs htdemucs_ft | 11.9 |
| Guitar/Piano | Demucs htdemucs_6s | — |

## Usage

### Python

```python
import replicate

output = replicate.run(
    "metaphysics0/bs-roformer",
    input={
        "audio": open("song.wav", "rb"),
        "stems": "vocals_instrumental",
        "output_format": "wav",
    },
)
# output = [vocals_url, instrumental_url]
```

### Async with webhook

```python
import replicate

prediction = replicate.predictions.create(
    model="metaphysics0/bs-roformer",
    input={
        "audio": "https://example.com/song.wav",
        "stems": "all_4",
        "output_format": "wav",
    },
    webhook="https://your-server.com/webhooks/replicate",
    webhook_events_filter=["completed"],
)
```

### cURL

```bash
curl -s -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "<version_id>",
    "input": {
      "audio": "https://example.com/song.wav",
      "stems": "vocals_instrumental",
      "output_format": "wav"
    }
  }'
```

## Inputs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | file | required | Audio file (WAV, MP3, FLAC, OGG, etc.) |
| `stems` | string | `vocals_instrumental` | `vocals_instrumental`, `all_4`, or `all_6` |
| `output_format` | string | `wav` | `wav`, `mp3`, or `flac` |
| `chunk_size` | int | `8` | Chunk size in seconds (4-30). Larger = better quality, more VRAM |
| `overlap` | int | `4` | Overlap between chunks (1-8). Higher = fewer boundary artifacts |
| `sample_rate` | int | `44100` | Output sample rate: 44100 or 48000 |

## References

- **Paper:** [Music Source Separation with Band-Split RoPE Transformer](https://arxiv.org/abs/2309.02612)
- **Weights:** [viperx/bs-roformer on HuggingFace](https://huggingface.co/viperx/bs-roformer)
- **BS-RoFormer implementation:** [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)
- **Demucs:** [facebookresearch/demucs](https://github.com/facebookresearch/demucs)

## License

MIT
