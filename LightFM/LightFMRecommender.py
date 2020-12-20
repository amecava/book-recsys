from Base.Recommender_utils import check_matrix
from Base.BaseRecommender import BaseRecommender
import numpy as np
import pandas as pd
import scipy.sparse as sps
from lightfm import LightFM


class LightFMRecommender(BaseRecommender):

    RECOMMENDER_NAME = "LightFM"

    def __init__(
        self,
        URM_train,
        ICM_train,
        no_components=1024,
        k=5,
        n=10,
        learning_schedule="adagrad",
        loss="logistic",
        learning_rate=0.05,
        rho=0.95,
        epsilon=1e-06,
        item_alpha=0.0,
        user_alpha=0.0,
        max_sampled=10,
        random_state=None,
    ):

        super(LightFMRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), "csr")
        self.ICM_train = check_matrix(ICM_train.copy(), "csr")

        # ICM_train_dense = pd.DataFrame(self.ICM_train.todense())
        # ICM_train_dense.index = ICM_train_dense.index.map(lambda x: item_mapper[str(x)])
        # self.ICM_train = sps.csr_matrix(ICM_train_dense.values)

        self.model = LightFM(
            no_components=no_components,
            k=k,
            n=n,
            learning_schedule=learning_schedule,
            loss=loss,
            learning_rate=learning_rate,
            rho=rho,
            epsilon=epsilon,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            max_sampled=max_sampled,
            random_state=random_state,
        )

    def fit(self, epochs=20):

        self.model.fit(self.URM_train, epochs=epochs)

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if items_to_compute is None:
            items_to_compute = list(range(self.n_items))

        item_scores = []

        for index, user_id in enumerate(user_id_array):

            item_scores.append(self.model.predict(user_id, items_to_compute).tolist())

        return np.array(item_scores)
