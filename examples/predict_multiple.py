#%%
import os

import pathlib
import pandas as pd

import tensorflow as tf

from mouffet.options.model_options import ModelOptions
from mouffet.utils.file import ensure_path_exists
from mouffet.utils.model_handler import ModelHandler

from biosoundnet.models import BioSoundNet
from biosoundnet.evaluation import predictions

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)


plots = [
    {
        "src_path": "/mnt/win/UMoncton/Doctorat/data/acoustic/field/2018/Plot1",
        "name": "test_bench",
    },
]

dest_root = pathlib.Path(
    "results/predict_multiple_bench_cpu"
)


model_opts = ModelOptions(
    {
        "model_dir": "resources/models/",
        "name": "BioSoundNet",
        "class": BioSoundNet,
        "batch_size": 64,
        "spectrogram_overlap": 0.95,
        "inference": True,
        "random_start": False,
        "ignore_parent_path": True,
    }
)

model = ModelHandler.load_model(model_opts)
overwrite = False

spec_opts = {"n_fft": 512, "n_mels": 32, "sample_rate": "original", "to_db": False}

infos_res = []
for plot in plots:
    dest_path = (
        pathlib.Path(dest_root)
        / plot["name"]
        / f"predictions_overlap{model_opts["spectrogram_overlap"]}.feather"
    )
    if not dest_path.exists() or overwrite:
        preds, infos = predictions.classify_elements(
            list(pathlib.Path(plot["src_path"]).glob("*.WAV")), model, spec_opts
        )
        preds.reset_index().to_feather(ensure_path_exists(dest_path, is_file=True))
        infos["plot"] = plot["name"]
        infos_res.append(infos)

infos_df = pd.DataFrame(infos_res)
infos_df.reset_index(drop=True).to_csv(dest_root / "predictions_stats.csv")
