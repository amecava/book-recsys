import pandas as pd


def _loadICM(ICM_path, header=True, separator=","):

    ICM_dataframe = pd.read_csv(
        filepath_or_buffer=ICM_path, sep=separator, header=header, dtype={0: str, 1: str, 2: float}
    )
    ICM_dataframe.columns = ["ItemID", "FeatureID", "Data"]

    return ICM_dataframe


def _loadURM(URM_path, header=None, separator=","):

    URM_dataframe = pd.read_csv(
        filepath_or_buffer=URM_path, sep=separator, header=header, dtype={0: str, 1: str, 2: float}
    )
    URM_dataframe.columns = ["UserID", "ItemID", "Data"]

    return URM_dataframe
