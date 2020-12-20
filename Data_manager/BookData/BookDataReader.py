import pandas as pd
from Data_manager.DataReader import DataReader
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.BookData._utils_bookdata_parser import _loadURM, _loadICM


class BookDataReader(DataReader):

    DATASET_SUBFOLDER = "BookData/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_all"]
    AVAILABLE_UCM = []

    IS_IMPLICIT = True

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):

        URM_path = "Data_manager/BookData/data_train.csv"
        ICM_path = "Data_manager/BookData/data_ICM_title_abstract.csv"

        self._print("Loading Interactions")
        URM_dataframe = _loadURM(URM_path, header=0, separator=",")

        self._print("Loading Item Features")
        ICM_dataframe = _loadICM(ICM_path, header=0, separator=",")

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_dataframe, "URM_all")
        dataset_manager.add_ICM(ICM_dataframe, "ICM_all")

        loaded_dataset = dataset_manager.generate_Dataset(
            dataset_name=self._get_dataset_name(), is_implicit=self.IS_IMPLICIT
        )

        self._print("Loading Complete")

        return loaded_dataset
