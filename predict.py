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
