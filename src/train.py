import os
import logging

# import tensorflow as tf
from mouffet.training.training_handler import TrainingHandler

from biosoundnet.data.audio_data_handler import AudioDataHandler

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.config.experimental.set_memory_growth(gpus[0], True)

# logging.basicConfig(level=logging.DEBUG)


trainer = TrainingHandler(
    opts_path="/home/vin/Doctorat/dev/biosoundnet/config/runs/run1/training_config.yaml",
    dh_class=AudioDataHandler,
)
trainer.train()
