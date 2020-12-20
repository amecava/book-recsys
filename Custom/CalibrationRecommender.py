from Base.Recommender_utils import check_matrix
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

import numpy as np


class CalibrationRecommender(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "CalibrationRecommender"

    def __init__(self, URM_train, Recommender):
        super(CalibrationRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), "csr")

        self.Recommender = Recommender

    def fit(self):

        item_weights = self.Recommender._compute_item_score(
            list(range(self.URM_train.shape[0]))
        )

        self.item_vmax_mean = []
        for i in range(self.n_items):
            item_scores = item_weights[:, i][item_weights[:, i] > 0]

            if len(item_scores) == 0:
                item_scores = [0]

            self.item_vmax_mean.append(
                {"max": np.max(item_scores), "mean": np.mean(item_scores)}
            )

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights = self.Recommender._compute_item_score(
            user_id_array, items_to_compute
        )

        for index, user_id in enumerate(user_id_array):
            user_scores = item_weights[index, :]

            for i, score in enumerate(user_scores):
                vmax = self.item_vmax_mean[i]["max"]
                mean = self.item_vmax_mean[i]["mean"]

                if mean != 0:
                    r = user_scores[i]
                    user_scores[i] = (
                        ((r / mean) * 0.5)
                        if r <= mean
                        else (0.5 + (r - mean) / (vmax - mean) * 0.5)
                        if r <= vmax
                        else 1
                    )

            item_weights[index, :] = user_scores

        return item_weights