# BS-RoFormer — Vocals/Instrumental Separation

State-of-the-art vocals/instrumental separation using [BS-RoFormer](https://arxiv.org/abs/2309.02612) (Band-Split RoPE Transformer) via [`audio-separator`](https://github.com/nomadkaraoke/python-audio-separator). Deployed on [Replicate](https://replicate.com/ryansmith/bs-roformer).

## What it does

Separates a song into **vocals** and **instrumental** stems (WAV output).

Uses the **viperx BS-RoFormer 1297** checkpoint — the best public model for vocal separation (SDR 12.97).

## Usage

### Python

```python
import replicate

output = replicate.run(
    "ryansmith/bs-roformer",
    input={"audio": open("song.wav", "rb")},
)
# output = [vocals_url, instrumental_url]
```

### Async with webhook

```python
import replicate

prediction = replicate.predictions.create(
    model="ryansmith/bs-roformer",
    input={"audio": "https://example.com/song.wav"},
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
      "audio": "https://example.com/song.wav"
    }
  }'
```

## Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | file | required | Audio file (WAV, MP3, FLAC, OGG, etc.) |

## Output

Returns a list of two WAV files: `[vocals.wav, instrumental.wav]`

## References

- **Paper:** [Music Source Separation with Band-Split RoPE Transformer](https://arxiv.org/abs/2309.02612)
- **Weights:** [viperx/bs-roformer on HuggingFace](https://huggingface.co/viperx/bs-roformer)
- **audio-separator:** [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)

## License

MIT
