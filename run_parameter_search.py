from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

import os, multiprocessing
from functools import partial


from Data_manager.BookData.BookDataReader import BookDataReader
from Data_manager.split_functions.split_train_validation_random_holdout import (
    split_train_in_two_percentage_global_sample,
)

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from ParameterTuning.run_parameter_search import runParameterSearch_Content


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    dataReader = BookDataReader()
    dataset = dataReader.load_data()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(
        dataset.get_URM_all(), train_percentage=0.8
    )
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(
        URM_train, train_percentage=0.8
    )

    ICM = dataset.get_ICM_from_name("ICM_all")

    output_folder_path = "Result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    collaborative_algorithm_list = [
        P3alphaRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        RP3betaRecommender,
    ]

    content_algorithm_list = [
        ItemKNNCBFRecommender,
    ]

    from Base.Evaluation.Evaluator import EvaluatorHoldout

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    runParameterSearch_Collaborative_partial = partial(
        runParameterSearch_Collaborative,
        URM_train=URM_train,
        metric_to_optimize="MAP",
        n_cases=1024,
        n_random_starts=32,
        evaluator_validation_earlystopping=evaluator_validation,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test,
        output_folder_path=output_folder_path,
        parallelizeKNN=False,
        save_model="no",
    )

    pool = multiprocessing.Pool(
        processes=int(multiprocessing.cpu_count()), maxtasksperchild=1
    )
    pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    runParameterSearch_Content_partial = partial(
        runParameterSearch_Content,
        URM_train=URM_train,
        ICM_object=ICM,
        ICM_name="ICM_all",
        metric_to_optimize="MAP",
        n_cases=1024,
        n_random_starts=32,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test,
        output_folder_path=output_folder_path,
        parallelizeKNN=False,
        save_model="no",
    )

    pool = multiprocessing.Pool(
        processes=int(multiprocessing.cpu_count()), maxtasksperchild=1
    )
    pool.map(runParameterSearch_Content_partial, content_algorithm_list)


if __name__ == "__main__":

    read_data_split_and_search()
