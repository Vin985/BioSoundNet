#%%

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from biosoundnet.evaluation import predictions
from biosoundnet.models import BioSoundNet
from mouffet.options.model_options import ModelOptions
from mouffet.utils import file_utils
from mouffet.utils.model_handler import ModelHandler

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


model_opts = ModelOptions(
    {
        "model_dir": "resources/models/",
        "name": "BioSoundNet",
        "class": BioSoundNet,
        "batch_size": 32,
        "spectrogram_overlap": 0.95,
        "inference": True,
        "random_start": False,
        "ignore_parent_path": True,
    }
)

model = ModelHandler.load_model(model_opts)


print(model.opts.opts)

spec_opts = {"n_fft": 512, "n_mels": 32, "sample_rate": "original", "to_db": False}

# file_path = Path("resources/audio/SESA.wav")

file_path = Path(
    "/mnt/win/UMoncton/Doctorat/data/acoustic/reference/Semipalmated Sandpiper.wav"
)
preds, info = predictions.classify_elements([file_path], model, spec_opts)

#%%
preds.plot("time", "activity")
dest_dir = Path("/mnt/win/UMoncton/Doctorat/dev/biosoundnet/examples/results")

plt.savefig(file_utils.ensure_path_exists(dest_dir) / "predict_single.png")
