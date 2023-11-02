#%%

from pathlib import Path

from mouffet import file_utils, config_utils
from biosoundnet.data.audio_data_handler import AudioDataHandler
from biosoundnet.evaluation import EVALUATORS
from biosoundnet.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)

from biosoundnet.applications.phenology.phenology_evaluator import PhenologyEvaluator

EVALUATORS.register_evaluator(PhenologyEvaluator)


#%%

models_dir = "../resources/models"

evaluation_config_path = "config/phenology/evaluation_config.yaml"

evaluation_config = file_utils.load_config(evaluation_config_path)


evaluation_config["models_list_dir"] = models_dir

evaluation_config = config_utils.get_models_conf(
    evaluation_config,
    # updates={
    #     "model_dir": "resources/models",
    #     # "ignore_parent_path": True,
    #     "spectrogram_overlap": 0.5,
    #     # "reclassify": True,
    # },
)


evaluator = SongDetectorEvaluationHandler(
    opts=evaluation_config, dh_class=AudioDataHandler
)

stats = evaluator.evaluate()
