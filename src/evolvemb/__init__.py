__version__ = "0.1.0"
from .base import EmbeddingInputModel, SimplePretrainedEmbeddings
from .embeddings import DummyEmbeddings, PretrainedEmbeddings, GlobalAvgEmbeddings, EvolvingEmbeddings
from .diachronic_utils import load_diachronic_dataset, compute_emb_snapshots, list_new_tokens, list_multiple_meanings_tokens, list_semantic_shift_tokens, plot_emb_over_time
