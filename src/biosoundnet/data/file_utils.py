from datetime import datetime

import pandas as pd

EXTRACT_INFO_FUNCS = {}


def recording_info_func(func):
    func_name = func.__name__
    info_type = func_name.replace("extract_info_", "")
    print(info_type)
    EXTRACT_INFO_FUNCS[info_type] = func
    return func


@recording_info_func
def extract_info_audiomoth2018(df, options):
    df["full_date"] = df.recording_id.path.stem.apply(
        lambda x: datetime.fromtimestamp(int(x, 16))
    )

    df["date"] = pd.to_datetime(df["full_date"].dt.strftime("%Y%m%d"))
    df["date_hour"] = pd.to_datetime(
        df["full_date"].dt.strftime("%Y%m%d_%H"), format="%Y%m%d_%H"
    )
    return df


@recording_info_func
def extract_info_biosoundnet(df, options):
    df[["site", "plot", "date", "time", "to_drop"]] = (
        df.recording_id.path.stem.str.split("_", expand=True)
    )
    df = df.assign(full_date=[str(x) + "_" + y for x, y in zip(df["date"], df["time"])])
    df["full_date"] = pd.to_datetime(df["full_date"], format="%Y-%m-%d_%H%M%S")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["date_hour"] = pd.to_datetime(
        df["full_date"].dt.strftime("%Y%m%d_%H"), format="%Y%m%d_%H"
    )
    df = df.drop(columns=["to_drop", "recording_id"])
    return df


@recording_info_func
def extract_info_audiomoth2019(df, options):
    df[["date", "time"]] = df.recording_id.path.stem.str.split("_", expand=True)
    df = df.assign(full_date=[str(x) + "_" + y for x, y in zip(df["date"], df["time"])])
    df["full_date"] = pd.to_datetime(df["full_date"], format="%Y%m%d_%H%M%S")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["date_hour"] = pd.to_datetime(
        df["full_date"].dt.strftime("%Y%m%d_%H"), format="%Y%m%d_%H"
    )
    return df


def extract_recording_info(df, info_type="biosoundnet", options=None):
    options = {} if not options else options
    func = EXTRACT_INFO_FUNCS.get(info_type)
    return func(df, options)
