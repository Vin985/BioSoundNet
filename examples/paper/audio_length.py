#%%

import pickle

data = pickle.load(
    open(
        "/mnt/win/UMoncton/Doctorat/data/dl_training/datasets/arctic_checked_final/test_metadata.pkl",
        "rb",
    )
)

audio_length = [x["duration"] for x in data]

print(min(audio_length))
print(max(audio_length))
