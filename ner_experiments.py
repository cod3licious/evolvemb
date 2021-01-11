import os
import shutil
import logging
import torch
import numpy as np
import copy as pycopy
from copy import deepcopy
from typing import List
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.datasets import DataLoader, CONLL_03, CONLL_03_GERMAN
from flair.trainers import ModelTrainer
from flair.training_utils import store_embeddings
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, ELMoEmbeddings, FlairEmbeddings, StackedEmbeddings, PooledFlairEmbeddings

from evolvemb import DummyEmbeddings, GlobalAvgEmbeddings, EvolvingEmbeddings, _preprocess_token

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# set this to True to reproduce the results from the paper exactly
# in the paper, the dataset is first embed in a way where the evolving embeddings for the test set don't see the data from the dev set
# otherwise the dataset is embedded in the first epoch during the sequence tagger training; while the documents are in the correct order
# (make sure you're using a recent flair version!!!), the test data is embedded after the dev data and the F1 score will be about 0.1 better
EMBEDFIRST = True


def preprocess_sentences(sentences: List[Sentence], to_lower=True, norm_num=True, num="0", copy=True):
    temp_sentences = pycopy.deepcopy(sentences) if copy else sentences
    for sentence in temp_sentences:
        for t in sentence:
            t.text = _preprocess_token(t.text, to_lower, norm_num, num)
    return temp_sentences


def preprocess_corpus(corpus, to_lower=True, norm_num=True, num="0", copy=True):
    temp_corpus = pycopy.copy(corpus) if copy else corpus
    temp_corpus._train = preprocess_sentences(temp_corpus.train, to_lower, norm_num, num, copy)
    temp_corpus._dev = preprocess_sentences(temp_corpus.dev, to_lower, norm_num, num, copy)
    temp_corpus._test = preprocess_sentences(temp_corpus.test, to_lower, norm_num, num, copy)
    return temp_corpus


def get_sentences(dataset="conll_en", to_lower=True):
    if dataset == "conll_de":
        corpus = CONLL_03_GERMAN("data")
    else:
        corpus = CONLL_03("data")
    corpus = preprocess_corpus(corpus, to_lower=to_lower, norm_num=True, num="0", copy=False)
    return [[t.text for t in sentence if t.text] for sentence in corpus._train]


def embed_dataset(dataset, embeddings):
    # make a batch loader
    data_loader = DataLoader(dataset, batch_size=32)
    # go through batches and embed
    for b_id, batch in enumerate(data_loader):
        embeddings.embed(batch)
        store_embeddings(batch, "cpu")


def embed_corpus_manually(corpus, embeddings):
    # embeddings can be a list of embeddings, then we want to use stacked embeddings
    if not isinstance(embeddings, list):
        embeddings = [embeddings]
    # hack to get the correct names for the stacked embeddings
    org_names = []
    for i in range(len(embeddings)):
        org_names.append(embeddings[i].name)
        if len(embeddings) > 1:
            embeddings[i].name = f"{str(i)}-{embeddings[i].name}"
    for i in range(len(embeddings)):
        print("[test_ner] embedding training corpus with %s (embedding_dim: %i)" % (embeddings[i].name, embeddings[i].embedding_length))
        # preembed all the sentences (--> important for evolving embeddings)
        embed_dataset(corpus.train, embeddings[i])
        # for evolving embeddings, embed dev and test independent so we don't compromise the evolving embeddings[i]
        print("[test_ner] embedding dev + test corpus")
        if isinstance(embeddings[i], EvolvingEmbeddings):
            local_emb = embeddings[i].local_embeddings
            embeddings[i].local_embeddings = None  # don't copy this part
            embeddings_dev = deepcopy(embeddings[i])
            embeddings_dev.local_embeddings = local_emb
            embed_dataset(corpus.dev, embeddings_dev)
            embeddings_test = deepcopy(embeddings[i])
            embeddings_test.local_embeddings = local_emb
            embed_dataset(corpus.test, embeddings_test)
            del local_emb, embeddings_dev, embeddings_test
        else:
            embed_dataset(corpus.dev, embeddings[i])
            embed_dataset(corpus.test, embeddings[i])
        # to save memory, we'll give the sequence tagger only a dummy embedding since we don't need to embed anything anymore
        embeddings[i] = DummyEmbeddings(embeddings[i].embedding_length, org_names[i])
    if len(embeddings) > 1:
        embeddings = StackedEmbeddings(embeddings)
    else:
        embeddings = embeddings[0]
    return embeddings


def test_ner(base_path, embeddings, dataset="conll_en"):
    # get the corpus
    if dataset == "conll_de":
        corpus = CONLL_03_GERMAN("data")
    else:
        corpus = CONLL_03("data")
    print(corpus)

    # set EMBEDFIRST to True to exactly reproduce the paper results with more complex adhoc embedding of the dataset
    if EMBEDFIRST:
        embeddings = embed_corpus_manually(corpus, embeddings)
    elif isinstance(embeddings, list):
        # embeddings can be a list of embeddings, then we want to use stacked embeddings
        embeddings = StackedEmbeddings(embeddings)

    # what tag do we want to predict?
    tag_type = "ner"

    # make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # initialize sequence tagger
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
    )

    # initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    trainer.train(
        base_path,
        learning_rate=0.1,
        mini_batch_size=32,
        max_epochs=150,
    )

    # clean up files
    shutil.copyfile(os.path.join(base_path, "training.log"), os.path.normpath(base_path) + ".txt")
    shutil.rmtree(base_path)
    # return tagger


def eval_pooledAkbik2019(use_fast=True, use_glove=True, fn="", dataset="conll_en"):
    """ run test_ner for the original pooled flair embeddings - make sure EMBEDFIRST = False!"""
    embeddings = []
    if use_glove:
        embeddings.append(WordEmbeddings('glove'))
    if use_fast:
        embeddings.extend([PooledFlairEmbeddings('news-forward-fast', pooling='mean'),
                           PooledFlairEmbeddings('news-backward-fast', pooling='mean')])
    else:
        embeddings.extend([PooledFlairEmbeddings('news-forward', pooling='mean'),
                           PooledFlairEmbeddings('news-backward', pooling='mean')])
    for i in range(len(embeddings)):
        # to make sure the embeddings aren't recomputed every epoch
        # also needs "if mode and not self.static_embeddings:" in train()
        embeddings[i].static_embeddings = True
    fname = "en_flair_pooled_fast%i_glove%i%s" % (int(use_fast), int(use_glove), fn)
    test_ner(os.path.join("results/ner_context/", fname), embeddings, dataset=dataset)


def get_transformer(name="bert-base-uncased"):
    """
    Get a transformer model

    Inputs:
        - name: one of "elmo", "flair", "bert-base-uncased", "roberta-base", "bert-base-german-cased"
    Returns:
        - (list of) embedding model(s)
    """
    if name == "elmo":
        embedding = ELMoEmbeddings("small", embedding_mode="average")
    elif name == "flair":
        embedding = [FlairEmbeddings('news-forward-fast'),  FlairEmbeddings('news-backward-fast')]
    else:
        # name: bert-base-uncased; roberta-base
        # bert-base-german-cased for german conll task
        embedding = TransformerWordEmbeddings(name, layers="all", use_scalar_mix=True, pooling_operation="mean", fine_tune=False)
    return embedding


def get_global_evolving(local_embeddings, evolving=True, alpha=None, dataset="conll_en"):
    """
    Get a global average or evolving embedding model

    Inputs:
        - local_embeddings: some model with an .embed() function that generates contextualized embeddings
        - evolving: whether to get evolving embeddings or global average embeddings
        - alpha: alpha for evolving embeddings
        - dataset: for global average embeddings based on which corpus the global embeddings should be generated
    Returns:
        - global/evolving embeddings model
        - name of model
    """
    if evolving:
        if alpha == "doc" or isinstance(alpha, tuple):
            if len(alpha) == 2:
                alpha = alpha[1]
            else:
                alpha = None
            emb = EvolvingEmbeddings(local_embeddings, alpha=alpha, reset_token="-DOCSTART-")
            name = "evolvedoc%s" % str(alpha)
        else:
            emb = EvolvingEmbeddings(local_embeddings, alpha=alpha, reset_token=None)
            name = "evolve%s" % str(alpha)
    else:
        emb = GlobalAvgEmbeddings(local_embeddings, get_sentences(dataset, True)).as_pretrained()
        name = "global"
    return emb, name


def eval_transformer(name="bert-base-uncased", fn="", dataset="conll_en"):
    """ run test_ner for a transformer model """
    embedding = get_transformer(name)
    fname = "%s_trans_local%s" % (name, fn)
    test_ner(os.path.join("results/ner_context/", fname), embedding, dataset=dataset)


def eval_evolving_transformer(name="bert-base-uncased", add_glove=False, stacked=False, evolving=True, alpha=None, fn="", dataset="conll_en"):
    """ run test_ner for evolving embeddings based on a transformer model """
    local_embeddings = get_transformer(name)
    if isinstance(local_embeddings, list):
        global_emb, gname = get_global_evolving(StackedEmbeddings(deepcopy(local_embeddings)), evolving=evolving, alpha=alpha, dataset=dataset)
    else:
        global_emb, gname = get_global_evolving(deepcopy(local_embeddings), evolving=evolving, alpha=alpha, dataset=dataset)
        local_embeddings = [local_embeddings]
    if add_glove:
        local_embeddings.append(WordEmbeddings('glove'))
        name = "glove+%s" % name
    if stacked:
        fname = "%s_%s_stacked_%s%s" % (dataset.split("_")[1], name.split("-")[0], gname, fn)
        test_ner(os.path.join("results/ner_context/", fname), local_embeddings + [global_emb], dataset=dataset)
    else:
        fname = "%s_%s_%s%s" % (dataset.split("_")[1], name.split("-")[0], gname, fn)
        test_ner(os.path.join("results/ner_context/", fname), global_emb, dataset=dataset)


if __name__ == '__main__':

    if False:
        # normal experiments with all models
        for transformer in ["bert-base-uncased", "roberta-base", "elmo", "flair", "bert-base-german-cased"]:
            if "german" in transformer:
                dataset = "conll_de"
            else:
                dataset = "conll_en"
            # check transformer on its own, i.e., local embeddings
            eval_transformer(name=transformer, dataset=dataset)
            # in combination with evolving embeddings
            for stacked in [False, True]:
                # eval_evolving_transformer(name=transformer, stacked=stacked, evolving=False, alpha=None, dataset=dataset)  # global non-evolving embeddings
                for alpha in [None, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.34, 0.5, ("doc", None), ("doc", 0.25), ("doc", 0.34), ("doc", 0.5)]:
                    eval_evolving_transformer(name=transformer, stacked=stacked, evolving=True, alpha=alpha, dataset=dataset)

    if False:
        # same as above just with multiple random seeds, i.e., to reproduce the experiments from the paper
        for transformer in ["bert-base-uncased", "roberta-base", "elmo", "flair", "bert-base-german-cased"]:
            if "german" in transformer:
                dataset = "conll_de"
            else:
                dataset = "conll_en"
            # check transformer on its own, i.e., local embeddings
            for i in range(3):
                torch.manual_seed(i)
                np.random.seed(i)
                eval_transformer(name=transformer, fn="_%i" % i, dataset=dataset)
                torch.cuda.empty_cache()
            # in combination with evolving embeddings
            for stacked in [False, True]:
                for alpha in [None, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.34, 0.5, ("doc", None), ("doc", 0.25), ("doc", 0.34), ("doc", 0.5)]:
                    for i in range(3):
                        torch.manual_seed(i)
                        np.random.seed(i)
                        eval_evolving_transformer(name=transformer, stacked=stacked, evolving=True, alpha=alpha, fn="_%i" % i, dataset=dataset)
                        torch.cuda.empty_cache()

    if True:
        # run flair with additional glove embeddings
        for stacked in [False, True]:
            for alpha in [None, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.34, 0.5, ("doc", None), ("doc", 0.25), ("doc", 0.34), ("doc", 0.5)]:
                for i in range(3):
                    torch.manual_seed(i)
                    np.random.seed(i)
                    eval_evolving_transformer(name="flair", add_glove=True, stacked=stacked, evolving=True, alpha=alpha, fn="_%i" % i, dataset="conll_en")
                    torch.cuda.empty_cache()

    if False:
        #global EMBEDFIRST  # from the way these embeddings are internally set up, they can't be used to embed the corpus beforehand
        EMBEDFIRST = False
        # use original pooled embeddings
        for use_fast in [True, False]:
            for use_glove in [False, True]:
                for i in range(3):
                    torch.manual_seed(i)
                    np.random.seed(i)
                    eval_pooledAkbik2019(use_fast=use_fast, use_glove=use_glove, fn="_%i" % i, dataset="conll_en")
                    torch.cuda.empty_cache()
