import logging
import sys
import tempfile
import traceback
from typing import List
from cog import BasePredictor, Input, Path

MODEL_NAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MODEL_DIR = "/src/weights"

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("predictor")


class Predictor(BasePredictor):

    def setup(self):
        """Load BS-RoFormer model into GPU."""
        try:
            log.info("setup: importing audio_separator")
            from audio_separator.separator import Separator
            log.info("setup: import successful")

            self.output_dir = tempfile.mkdtemp()
            log.info("setup: initializing Separator")
            self.separator = Separator(
                log_level=logging.INFO,
                model_file_dir=MODEL_DIR,
                output_dir=self.output_dir,
                output_format="WAV",
                mdxc_params={
                    "segment_size": 256,
                    "override_model_segment_size": False,
                    "batch_size": 1,
                    "overlap": 8,
                    "pitch_shift": 0,
                },
            )
            log.info("setup: loading model %s", MODEL_NAME)
            self.separator.load_model(model_filename=MODEL_NAME)
            log.info("setup: done")
        except Exception:
            log.error("setup failed:\n%s", traceback.format_exc())
            raise

    def predict(
        self,
        audio: Path = Input(
            description="Audio file to separate (WAV, MP3, FLAC, OGG, etc.)"
        ),
    ) -> List[Path]:
        """Separate audio into vocals and instrumental stems."""
        output_files = self.separator.separate(str(audio))
        return [Path(f) for f in output_files]
