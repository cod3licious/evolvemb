import re
import unicodedata
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _preprocess_token(text, to_lower=True, norm_num=True, num="0"):
    # no non-ascii characters
    nfkd_form = unicodedata.normalize("NFKD", text)
    text = nfkd_form.encode("ASCII", "ignore").decode("ASCII")
    if to_lower:
        # all lower case
        text = text.lower()
    if norm_num:
        # normalize numbers
        text = re.sub(r"\d", num, text)
    return text.strip()


def _generate_preprocessed_tokens(text):
    yield text                                    # original text
    t = _preprocess_token(text, False, False)     # without weird characters
    yield t
    yield t.lower()                               # lower case
    yield t.title()                               # title case
    if not t.isalpha():                           # different number normalizations
        yield re.sub(r"\d", "0", t.lower())
        yield re.sub(r"\d", "#", t.lower())
        yield re.sub(r"\d", "0", t)
        yield re.sub(r"\d", "#", t)


class EmbeddingInputModel:

    def __init__(self, sentences, min_freq=1, n_tokens=None, verbose=1):
        """
        Initialize the EmbeddingInputModel by letting it check the sentences to generate
        a mapping from token to index and vice versa (used e.g. to index the all the embeddings here)

        :param sentences: a list of lists of words
        :param min_freq: how often a token needs to occur to be considered as a feature
        :param n_tokens: how many tokens to keep at most (might be less depending on min_freq; default: all)
        :param verbose: whether to generate warnings (default: 1)
        """
        self.min_freq = min_freq
        self.token_counts = Counter(t for sentence in sentences for t in sentence)
        self.index2token = [t for t, c in self.token_counts.most_common(n_tokens) if c >= min_freq]
        if not self.index2token and verbose:
            print("[EmbeddingInputModel] WARNING: no tokens with frequency >= %i" % min_freq)
        self.token2index = {t: i for i, t in enumerate(self.index2token)}

    @property
    def n_tokens(self) -> int:
        return len(self.index2token)

    def update_index(self, sentence):
        # possibly add the tokens from the new sentence to the index
        for token in set(sentence):
            if token not in self.token2index:
                self.token2index[token] = len(self.index2token)
                self.index2token.append(token)

    def get_token(self, token_text, default=None):
        # get the closest matching token in our vocab
        if token_text is None:
            return default
        for t in _generate_preprocessed_tokens(token_text):
            if t in self.token2index:
                return t
        return default

    def get_index(self, token_text, default=-1):
        # get the index of the closest matching token in our vocab
        if token_text is None:
            return default
        for t in _generate_preprocessed_tokens(token_text):
            if t in self.token2index:
                return self.token2index[t]
        return default


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
