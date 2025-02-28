import math
import time
import traceback
from pathlib import Path

import mouffet.utils.common as common_utils
import numpy as np
import pandas as pd

from ..data import audio_utils
from ..training import SpectrogramSampler

# import tracemalloc


def classify_elements(elements, model, spec_opts=None):
    infos = {}
    res = []
    total_audio_duration = 0
    test_sampler = SpectrogramSampler(model.opts, randomise=False, balanced=False)
    test_sampler.opts["random_start"] = False

    common_utils.print_info(
        "Classifying {} elements with model with options: {} and sampler with options {}".format(
            len(elements), model.opts, test_sampler.opts
        )
    )
    infos["n_files"] = len(elements)

    to_classify = None
    lengths = []
    min_duration = model.opts.get("classify_min_duration", 0)
    dur = 0

    # * Timing statistics
    load_times = []
    preprocess_times = []
    classify_times = []
    classify_times_pm = []
    total_times = []
    classify_props = []
    start = time.time()
    to_load = False

    for i, element in enumerate(elements):
        print("Classifying element {}/{}".format(i, len(elements)))
        try:
            start_file = time.time()
            preprocess_start = time.time()
            load_start = time.time()
            if isinstance(element, Path) or isinstance(element, str):
                to_load = True
                if not spec_opts:
                    raise AttributeError(
                        (
                            "Error trying to classify {}: spec_opts arguments "
                            + "is missing. Please specify a spec_opts arguments "
                            + "while classifying elements from a file path."
                        ).format(element)
                    )
                try:
                    (spec, metadata, _) = audio_utils.load_audio_data(
                        element, spec_opts
                    )

                except Exception:
                    common_utils.print_error(traceback.format_exc())
                    with open("loading_error.log", "a", encoding="utf8") as error_log:
                        error_log.write(str(element) + "\n")
                    continue

            else:
                spec, metadata = element

            load_time = load_start - time.time()
            duration = metadata["duration"]
            total_audio_duration += duration
            if duration < min_duration:
                if to_classify is None:
                    to_classify = np.zeros((spec.shape[0], 0))
                    lengths = [spec.shape[1]]
                    dur = 0
                else:
                    lengths.append(duration)
                padding = np.zeros(
                    (spec.shape[0], math.ceil(spec.shape[1] / duration) * 3), np.float32
                )
                to_classify = np.hstack([to_classify, spec, padding])
                dur += duration + 3
                if dur < min_duration:
                    continue
                metadata["duration"] = dur
            else:
                # info["duration"] = duration
                to_classify = spec
                # info["duration"] = 30
                # idx = math.ceil(spec.shape[1] / duration * 30)
                # to_classify = spec[:, 0:idx]
            if to_load:
                to_classify = audio_utils.modify_spectrogram(
                    to_classify,
                    model.opts,
                    resize_width=audio_utils.get_resize_width(
                        model.opts.get("pixels_per_second", 100), duration
                    ),
                )
            preprocess_time = time.time() - preprocess_start
            start_classify = time.time()
            res_df = classify_element(model, (to_classify, metadata), test_sampler)
            classify_time = time.time() - start_classify
            classify_time_pm = classify_time * 60 / duration
            # plt = res_df.plot("time", "activity")
            # fig = plt.get_figure()
            # fig.savefig("output.png")
            to_classify = None
            res.append(res_df)
            total_time = time.time() - start_file
            classify_prop = classify_time / total_time
            print(f"File done in {total_time}")

            load_times.append(load_time)
            preprocess_times.append(preprocess_time)
            classify_times.append(classify_time)
            classify_times_pm.append(classify_time_pm)
            total_times.append(total_time)
            classify_props.append(classify_prop)
            # snapshot2 = tracemalloc.take_snapshot()
            # top_stats = snapshot2.compare_to(snapshot1, "lineno")
            # print("[ Top 10 differences ]")
            # for stat in top_stats[:10]:
            #     print(stat)
            # snapshot1 = snapshot2
        except Exception:
            common_utils.print_error(traceback.format_exc())

    end = time.time()

    infos["global_duration"] = round(end - start, 2)
    infos["total_audio_duration"] = round(total_audio_duration, 2)
    infos["average_time_per_min"] = round(
        infos["global_duration"] / (total_audio_duration / 60), 2
    )

    if infos["n_files"] > 0:
        infos["average_time_per_file"] = round(
            infos["global_duration"] / infos["n_files"], 2
        )

    infos["spectrogram_overlap"] = test_sampler.opts["overlap"]

    infos["average_classification_time"] = round(np.mean(classify_times), 2)
    infos["average_classification_time_per_minute"] = round(
        np.mean(classify_times_pm), 2
    )
    infos["average_total_time"] = round(np.mean(total_times), 2)
    infos["average_loading_time"] = round(round(np.mean(load_times)), 2)
    infos["average_preprocessing_time"] = round(round(np.mean(preprocess_times)), 2)
    infos["average_classify_prop"] = round(np.mean(classify_props), 2)

    preds = pd.concat(res)
    preds = preds.astype({"recording_path": "category"})
    return preds, infos


def classify_element(model, data, sampler):
    spectrogram, info = data
    preds = model.predict_spectrogram((spectrogram, sampler))
    len_in_s = info["duration"]
    timeseq = np.linspace(0, len_in_s, preds.shape[0])
    res_df = pd.DataFrame(
        {
            "recording_path": str(info["file_path"]),
            "time": timeseq,
            "activity": preds,
        }
    )
    return res_df


def smooth_predictions(preds, model_opts):
    factor = model_opts.get("smooth_factor", 3)
    if factor:
        roll = preds["activity"].rolling(factor, center=True)
        preds.loc[:, "activity"] = roll.mean()
    return preds
