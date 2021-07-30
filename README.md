# Continuously Evolving Embeddings

This repository contains code to compute continuously evolving word embeddings as weighted running averages of contextualized embeddings (e.g. computed by a transformer model such as BERT). Specifically, these embeddings can be used as continuous diachronic embeddings to study the semantic shift and usage change of words over time as described in our [paper](10.18653/v1/2021.acl-demo.35).


## Library Usage

The code is written for Python 3. Please make sure you have all the dependencies from the `requirements.txt` file installed (especially `flair>=0.8.0`).

### Computing Continuously Evolving Embeddings

The main code to compute the evolving embeddings can be found in the file `evolvemb/embeddings.py`. The evolving embeddings are built on top of Flair Embeddings. They can be used as follows:

```python
from flair.embeddings import TransformerWordEmbeddings
from evolvemb import EvolvingEmbeddings

# the contextualized embeddings from which the weighted running averages should be computed
local_emb = TransformerWordEmbeddings("bert-base-uncased", layers="all", layer_mean=True, subtoken_pooling="mean", fine_tune=False)
# initialize the continuously evolving embeddings on top of the BERT embeddings with some alpha for the weighted average
evolving_emb = EvolvingEmbeddings(local_emb, alpha=0.1)

# ## basic, direct usage (--> see also diachronic embedding example)
# the corpus from which the embeddings should be learned: list of lists of words
sentences = [["This", "is", "one", "sentence", "."], ["This", "is", "another", "sentence", "."]]
# learn embeddings
for s in sentences:
    evolving_emb.update_evolving_embeddings(s)
# get embedding vector for a word of interest as a numpy array
emb_vector = evolving_emb["sentence"]

# ## working with Flair data structures
from flair.data import Sentence
# make a flair sentence
sentence = Sentence('I love Berlin .')
# embed sentence with evolving embeddings
evolving_emb.embed(sentence)
```

If you need to save the final embeddings without the dependency on the flair library, you can have a look at the `SimplePretrainedEmbeddings` class in `evolvemb/base.py` and the diachronic embedding example.


### Exploring Diachronic Embeddings

The following steps need to be executed to explore diachronic embeddings created from your own dataset in a local dash web app with interactive graphics (see [screencast](https://youtu.be/ltF67J-la7I) for a demonstration). See `data/nytimes_dataset_excerpt.txt` for a preview of how the full `nytimes_dataset.txt` file used in the examples looks like.

1. [Optional (but advised)] **Fine-tune Transformer Model**: especially if your data is very different from any of the available pre-trained HuggingFace models, consider fine-tuning a related model checkpoint on your dataset. The `finetune_huggingface.ipynb` notebook contains code for how we did this on our NYTimes dataset.

2. **Dataset Preparation & Computation of Continuously Evolving Embedding Snapshots**: see `nytimes_diachronic.ipynb` for an example of how to do this.
The embedding snapshots can be computed with the function `compute_emb_snapshots` from `evolvemb`. The function requires as input:
    - `sentences`: a list of sentences, where a sentence is represented as a list of words (possibly preprocessed, i.e., strings will be passed as is to the transformer model). The sentences have to be in chronological order (i.e., oldest first), corresponding to the dates (see below).
    - `dates`: a list of dates (as strings) of the same length as `sentences`, i.e., each date corresponds to the respective sentence in the previous list. The date strings should be written in a format that can easily be compared, e.g., `'%Y-%m-%d'` (-> `'2021-02-16'`). If your dataset is in the same format as our original dataset (see `data/nytimes_dataset_excerpt.txt`), then you can also use the function `load_diachronic_dataset` to read in your data and generate the `sentences` and `dates` lists.
    - `snapshots`: a list of strings with dates after which the snapshot should be saved; in the same format as the strings in `dates`. For example, `snapshots=['2020-01-31', '2020-02-31', '2020-03-31']` will result in 3 embedding snapshots that are saved at the end of Jan/Feb/Mar 2020 respectively. It doesn't matter that the snapshots contain impossible dates, e.g., the snapshot for `'2020-02-31'` is simply taken before the first sentence with the date `'2020-03-01'` is processed.
    - `local_emb_name`: this will be passed as the checkpoint name to the transformer model. By default, random embeddings are used, which can be helpful to quickly test your code without applying the transformer to all your sentences.

    The `compute_emb_snapshots` function then returns a dictionary with `{'snapshot-date': SimplePretrainedEmbeddings}` where the snapshot dates consist of the actual dates where the snapshots were taken (e.g. `'2020-02-28'` instead of `'2020-02-31'`) and the embeddings themselves are `SimplePretrainedEmbeddings` from `evolvemb/base.py`, which is a convenience format to easily access embeddings in a numpy array with keys, e.g., `snapshot_emb['2020-02-28']['banana']` could give you the snapshot of the embedding for the word 'banana' at the end of February 2020. <br>
    To use the computed embedding snapshots in the next step (web app with interactive plots), use pickle to save them as `snapshot_emb.pkl` in the same folder as `app.py`, i.e., `pickle.dump(snapshot_emb, open("snapshot_emb.pkl", "wb"), -1)`.

3. **Interactive Dash App**: Once you've computed and saved the embedding snapshots as `snapshot_emb.pkl`, the interactive dash app can be started by running `$ python app.py` and opening a browser at http://127.0.0.1:8050/, where you can now explore the plots for different words in your dataset. Alternatively, the same plots can also be viewed in a Jupyter Notebook (see `nytimes_diachronic.ipynb` for an example).

The web app and plots can also easily be created using other diachronic embeddings by simply wrapping the existing embedding matrices in a `SimplePretrainedEmbeddings` or `PretrainedEmbeddings` object (e.g. `gensim` `KeyedVectors`) when constructing the `snapshot_emb` dictionary; see `nytimes_diachronic_gensim.ipynb` for an example based on [Kim et. al 2014](https://arxiv.org/pdf/1405.3515.pdf).

## Reproducing the Results from the Paper

In addition to the `evolvemb` library, this repository also contains all necessary code to reproduce the results from our paper: first the New York Times article snippets need to be downloaded from the NYTimes API. For this the script `nytimes_make_dataset.py` can be used, which requires an API key in a file `nytimes_apikey.txt` and saves the final dataset at `data/nytimes_dataset.txt` (see `data/nytimes_dataset_excerpt.txt` for a preview of how the full dataset is structured). Once the dataset was created, the experiments can be executed in the notebook `nytimes_diachronic.ipynb` (and `nytimes_diachronic_gensim.ipynb`). Please note that the date given for some NYTimes article snippets is noisy, because an article can be updated again at a much later date without this being reflected in the data obtained from the API (this becomes noticeable, for example, when looking at the plots for 'coronavirus').


## Citing the Paper

If any of this code was helpful for you, please consider citing the [paper](10.18653/v1/2021.acl-demo.35):

```
@inproceedings{horn2021exploring,
    title = "Exploring Word Usage Change with Continuously Evolving Embeddings",
    author = "Franziska Horn",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-demo.35",
    doi = "10.18653/v1/2021.acl-demo.35",
    pages = "290--297",
}
```
