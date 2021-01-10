import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from evolvemb import EmbeddingInputModel


class SimplePretrainedEmbeddings():

    def __init__(self, embeddings: np.array, input_model: EmbeddingInputModel, include_oov=False):
        """
        Everything that is needed to embed words with the given pretrained embeddings.

        :param embeddings: either pretrained gensim embeddings or a matrix with n_tokens x embedding_dim
        :param input_model: input_model model
        :param include_oov: whether an OOV embedding at index -1 needs to be created (default: False)
        """
        self.input_model = input_model
        # possibly add OOV embedding at -1
        if include_oov:
            self.embeddings = np.vstack([embeddings, np.zeros(embeddings.shape[1])])
        else:
            self.embeddings = embeddings

    @property
    def embedding_length(self) -> int:
        return self.embeddings.shape[1]

    def __contains__(self, token):
        return self.input_model.get_token(token) is not None

    def __getitem__(self, token):
        # get embedding for a single token (text) as a numpy array with -1 = OOV
        return self.embeddings[self.input_model.get_index(token)]

    def get_nneighbors(self, token, k=5, include_simscore=True):
        """
        Inputs:
            - token: token for which to compute the nearest neighbors
            - k: how many nearest neighbors to find (default: 5)
            - include_simscore: whether the similarity score of the token should be included in the results (default: True)
        Returns:
            - nearest_neighbors: list of k tokens that are most similar to the given token (+ similarity score)
        """
        if self.input_model.get_token(token) is None:
            print("not in vocabulary:", token)
            return []
        # nearest neighbors idx based on cosine similarity of token to other embeddings
        sims = cosine_similarity(self.embeddings, self.__getitem__(token).reshape(1, -1)).flatten()
        nn_idx = np.argsort(sims)[::-1]
        # make sure it's not the token itself or OOV
        nn_tokens = []
        for i in nn_idx:
            if i < self.embeddings.shape[0] - 1:
                t = self.input_model.index2token[i]
                if t != token:
                    nn_tokens.append((t, sims[i]) if include_simscore else t)
            if len(nn_tokens) >= k:
                break
        return nn_tokens
