# %%


from pathlib import Path

import numpy as np
import pandas as pd
from biosoundnet.applications.phenology.phenology_evaluator import PhenologyEvaluator

src_file = Path(
    "/mnt/win/UMoncton/Doctorat/results/birdnet/results/BirdNET_CombinedTable.csv"
)
biosoundnet_preds = pd.read_feather(
    "/mnt/win/UMoncton/Doctorat/dev/BioSoundNet/examples/paper/results/predictions/original_mel32_512_None_None/full_summer_final_BioSoundNet_v1_overlap0.75.feather"
)

time_seq = biosoundnet_preds[
    biosoundnet_preds["recording_path"] == biosoundnet_preds["recording_path"].iloc[0]
]["time"]

results_df = pd.read_csv(src_file).rename(
    columns={
        "Start (s)": "start",
        "End (s)": "end",
        "Scientific name": "scientific_name",
        "Common name": "common_name",
        "Confidence": "confidence",
        "File": "file",
    }
)


classes = [
    # "Engine",
    "Brant",
    # "Barred Owl",
    # "Fireworks",
    # "Human vocal",
    # "Coyote",
    "Ruddy Shelduck",
    # "Human non-vocal",
    "Greater Sage-Grouse",
    "Wood Lark",
    "Snow Bunting",
    "Cackling Goose",
    # "Black Skimmer",
    # "Savi's Warbler",
    "Snow Goose",
    "Red-throated Loon",
    # "Black-crowned Night-Heron",
    "Lapland Longspur",
    "American Golden-Plover",
    # "Southern Screamer",
    "Kelp Gull",
    "Greater White-fronted Goose",
    "Tundra Swan",
    "Terek Sandpiper",
    # "White-browed Antbird",
    # "Great Grebe",
    # "Cinnamon Attila",
    "Purple Sandpiper",
    "Semipalmated Sandpiper",
    # "Golden-throated Barbet",
    # "Great Bittern",
    # "Red-legged Seriema",
    # "Eurasian Coot",
    "Black-bellied Plover",
    "Common Crane",
    "Red Phalarope",
    "Hooded Grebe",
    # "Eurasian Scops-Owl",
    # "European Pied Flycatcher",
    # "Little Bittern",
    # "Lesser Nighthawk",
    # "Black Scoter",
    "White-rumped Sandpiper",
    # "Elf Owl",
    # "Giant Snipe",
    # "Roseate Spoonbill",
    "Ruddy Turnstone",
    # "Cattle Egret",
    # "Eastern Screech-Owl",
    "European Golden-Plover",
    "Long-tailed Duck",
    # "Paraguayan Snipe",
    # "Eurasian Curlew",
    # "Brown Quail",
    # "White's Thrush",
    # "Peruvian Screech-Owl",
    # "Gray Heron",
    "Herring Gull",
    # "Little Owl",
    # "Black-breasted Parrotbill",
    # "Scaled Dove",
    # "Eurasian Bullfinch",
    # "Northern Pygmy-Owl",
    "Solitary Sandpiper",
    # "Little Crake",
    "Common Tern",
    "Arctic Tern",
    # "Swainson's Flycatcher",
    # "Black-billed Cuckoo",
    # "Black-throated Antshrike",
    "Yellow Tyrannulet",
    "Wood Sandpiper",
    "Pacific Loon",
    # "Burrowing Owl",
    # "Gray-cowled Wood-Rail",
    # "Sage Thrasher",
    # "Crested Lark",
    "Sabine's Gull",
    # "Large-billed Lark",
    "Willow Ptarmigan",
    # "Tawny Owl",
    # "Booted Warbler",
    # "Least Bittern",
    # "Siberian Rubythroat",
    # "European Stonechat",
    # "Eurasian Skylark",
    "Canada Goose",
    # "Eurasian Nightjar",
    # "Russet-crowned Crake",
    # "Mistletoe Tyrannulet",
    # "Mangrove Cuckoo",
    # "Great Jacamar",
    # # "Lesser Ground-Cuckoo",
    # "Imperial Snipe",
    # "Juniper Titmouse",
    # "Human whistle",
]

results_df = results_df[results_df["common_name"].isin(classes)]
# %%


def compile_results(df, exclude_classes, time_seq, confidence_threshold=-1):
    tmp = df[~df["common_name"].isin(exclude_classes)]
    if confidence_threshold > 0:
        tmp = df[df["confidence"] > confidence_threshold]
    res = np.zeros(len(time_seq))
    for _, annot in tmp.iterrows():
        start_point = int(float(annot["start"]))
        end_point = int(float(annot["end"]))
        res[start_point:end_point] = 1
    # res[res == 1] = 0
    # res[res > 1] = 1
    return pd.DataFrame(
        {"recording_id": df.name, "events": pd.Series(res), "time": time_seq}
    )


# %%


linear_presence = (
    results_df.groupby("file")
    .apply(
        compile_results,
        exclude_classes=[
            # "Human vocal",
            # "Engine",
            # "Fireworks",
            # "Human non-vocal",
            # "Human whistle",
        ],
        time_seq=time_seq,
        confidence_threshold=0.8,
    )
    .reset_index(drop=True)
)
linear_presence

# %%

evaluator = PhenologyEvaluator()


trends = evaluator.get_daily_trends(linear_presence, {"method": "direct"})
trends["trends_df"].plot("date", "trend")
