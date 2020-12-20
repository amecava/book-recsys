from Base.Recommender_utils import check_matrix
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

from Base.NonPersonalizedRecommender import TopPop


class ColdUsersTopPopRecommender(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "ColdUsersTopPopRecommender"

    def __init__(self, URM_train, Recommender):
        super(ColdUsersTopPopRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), "csr")

        self.Recommender = Recommender

        self.topPop = TopPop(URM_train)

    def fit(self):

        self.topPop.fit()

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights = self.Recommender._compute_item_score(user_id_array, items_to_compute)

        for index, user_id in enumerate(user_id_array):
            if self._get_cold_user_mask()[user_id]:
                item_weights[index] += self.topPop._compute_item_score([user_id], items_to_compute)[0]

        return item_weights
