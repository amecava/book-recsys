from tqdm import tqdm
from datetime import datetime

from Base.DataIO import DataIO
from Data_manager.BookData.BookDataReader import BookDataReader

TARGET_path = "Data_manager/BookData/data_target_users_test.csv"
OUTPUT_path = "Result_experiments/"


def load_recommender(recommender_class, URM, ICM=None, metadata_args=""):
    metadata_path = recommender_class.RECOMMENDER_NAME + metadata_args + "_metadata.zip"
    metadata = data_loader.load_data(metadata_path)
    best_parameters = metadata["hyperparameters_best"]

    if ICM is None:
        recommender = recommender_class(URM)
    else:
        recommender = recommender_class(URM, ICM)

    recommender.fit(**best_parameters)

    return recommender


def set_recommender(dict, recommender, key=None):
    if key is None:
        key = recommender.RECOMMENDER_NAME

    dict[key] = {}
    dict[key]["model"] = recommender


def create_csv(results, users_column, items_column, results_directory="./"):
    csv_filename = "results_"
    csv_filename += datetime.now().strftime("%b%d_%H-%M-%S") + ".csv"
    with open(csv_filename, "w") as file:
        file.write(users_column + "," + items_column + "\n")
        for key, value in results.items():
            file.write(str(key) + ",")
            first = True
            for prediction in value:
                if not first:
                    file.write(" ")
                first = False
                file.write(str(prediction))
            file.write("\n")


if __name__ == "__main__":

    dataset_object = BookDataReader()
    dataset = dataset_object.load_data()

    URM_all = dataset.get_URM_from_name("URM_all")
    ICM_all = dataset.get_ICM_from_name("ICM_all")

    URM_train = URM_all

    data_loader = DataIO(folder_path=OUTPUT_path)

    mapper = {}
    for key, value in dataset.get_item_original_ID_to_index_mapper().items():
        mapper[value] = key

    dict = {}

    ##########################################

    from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

    recommender = ItemKNNCFRecommender(URM_train)
    recommender.fit(
        topK=337, shrink=722, similarity="asymmetric", asymmetric_alpha=0.11770705488555902, feature_weighting="TF-IDF"
    )

    set_recommender(dict, recommender)

    from GraphBased.RP3betaRecommender import RP3betaRecommender

    recommender = RP3betaRecommender(URM_train)
    recommender.fit(topK=333, alpha=0.40733936199594367, implicit=True, normalize_similarity=False)

    set_recommender(dict, recommender)

    from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

    recommender = ItemKNNCBFRecommender(URM_train, ICM_all)
    recommender.fit(topK=142, shrink=60, similarity="jaccard", normalize=False)

    set_recommender(dict, recommender)

    ##########################################

    import pandas as pd

    user_id_array = pd.DataFrame(URM_train.todense()).index.tolist()
    len(user_id_array)

    recommender = dict[ItemKNNCBFRecommender.RECOMMENDER_NAME]["model"]

    recommended_items = recommender.recommend(user_id_array, cutoff=6)

    URM_temp = URM_train.copy()

    for user_id, item_list in enumerate(tqdm(recommended_items)):
        for item_index in item_list:
            URM_temp[user_id, item_index] = 0.05

    from Hybrid.PipelingRecommender import PipelingRecommender

    baseline = dict[ItemKNNCFRecommender.RECOMMENDER_NAME]["model"]

    temporary = ItemKNNCFRecommender(URM_temp)
    temporary.fit(
        topK=337, shrink=722, similarity="asymmetric", asymmetric_alpha=0.11770705488555902, feature_weighting="TF-IDF"
    )

    recommender = PipelingRecommender(URM_train, temporary)

    set_recommender(dict, recommender, key="Pipeling_ItemKNNCFRecommender")

    ##########################################

    ##########################################

    recommender = dict["Pipeling_RP3betaRecommender_ItemKNNCFRecommender_ItemKNNCBFRecommender"]["model"]

    ##########################################

    target_file = open(TARGET_path)
    row_list = list(target_file)[1:]
    target_users_list = []
    for row in row_list:
        target_users_list.append(int(row))

    results = {}
    for user in tqdm(target_users_list):
        ranking = recommender.recommend(user, cutoff=10)

        results[user] = []
        for index in ranking:
            results[user].append(int(mapper[index]))

    create_csv(
        results,
        users_column="user_id",
        items_column="item_list",
    )
