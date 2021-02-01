# Continuously Evolving Embeddings

This repository contains the code to reproduce the results of our paper submitted to ACL 2021.


### Continuously Evolving Embeddings

The main code to compute the evolving embeddings can be found in the file `evolvemb.py`. The evolving embeddings are built on top of Flair Embeddings. They can be used as follows:

```python
from flair.embeddings import TransformerWordEmbeddings
from evolvemb import EvolvingEmbeddings

# the contextualized embeddings from which the weighted running averages should be computed
local_emb = TransformerWordEmbeddings("bert-base-uncased", layers="all", use_scalar_mix=True, pooling_operation="mean", fine_tune=False)
# initialize the continuously evolving embeddings on top of the BERT embeddings with some alpha for the weighted average
evolving_emb = EvolvingEmbeddings(local_emb, alpha=0.1)

# ## basic, direct usage (--> see also diachronic embedding example)
# the corpus from which the embeddings should be learned: list of lists of words
sentences = [["This", "is", "one", "sentence", "."], ["This", "is", "another", "sentence", "."]]
# learn embeddings
for s in enumerate(sentences):
    evolving_emb.update_evolving_embeddings(s)
# get embedding vector for a word of interest as a numpy array
emb_vector = evolving_emb["sentence"]

# ## working with Flair data structures (--> see also NER example)
from flair.data import Sentence
# make a flair sentence
sentence = Sentence('I love Berlin .')
# embed sentence with evolving embeddings
evolving_emb.embed(sentence)
```

If you need to save a snapshot of the embeddings without the dependence on the flair library, you can have a look at `emb_noflair.py` and the diachronic embedding example. The evolving embeddings will also be integrated into the flair library directly and the code will be maintained there.


### Diachronic Embeddings

To recreate the diachronic embedding results, first the article snippets need to be downloaded from the NYTimes API. For this the script `nytimes_make_dataset.py` can be used, which requires an API key in a file `nytimes_apikey.txt`. Once the dataset was created, the experiments can be executed in the notebook `nytimes_diachronic.ipynb`.


### Named Entity Recognition (Supplementary Materials)

- `ner_experiments.py` contains the code to reproduce the NER results from the paper; it requires the files from the English and German CoNLL NER tasks in the respective folder.
- `ner_paper_plots.py` contains the code to recreate the figures from the paper.
