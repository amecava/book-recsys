from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender


class ItemKNNSimilarityHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"

    def __init__(self, URM_train, Similarity_1, Similarity_2, sparse_weights=True):
        super(ItemKNNSimilarityHybridRecommender, self).__init__(URM_train)

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError(
                "ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                    Similarity_1.shape, Similarity_2.shape
                )
            )

        self.Similarity_1 = check_matrix(Similarity_1.copy(), "csr")
        self.Similarity_2 = check_matrix(Similarity_2.copy(), "csr")

    def fit(self, topK=100, alpha=0.5):

        self.topK = topK
        self.alpha = alpha

        W = self.Similarity_1 * self.alpha + self.Similarity_2 * (1 - self.alpha)
        self.W_sparse = similarityMatrixTopK(W, k=self.topK).tocsr()