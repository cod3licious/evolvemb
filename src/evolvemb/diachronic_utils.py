# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import KernelPCA
from flair.embeddings import TransformerWordEmbeddings

from .base import SimplePretrainedEmbeddings
from .embeddings import EvolvingEmbeddings, DummyEmbeddings


def compute_emb_snapshots(sentences, dates, snapshots, local_emb_name="dummy", min_freq=100, n_tokens=10000):
    """
    Compute embedding snapshot from the given sentences.

    Inputs:
        - sentences [list: [list: str]]: list of sentences (as lists of words) in chronological order
        - dates [list: str]: list of the same length as sentences with corresponding dates; date format needs to be compatible with snapshot dates
        - snapshots [list: str]: list of dates (as strings) at which points the snapshots should be taken
        - local_emb_name [str]: which contextualized embeddings should be used when computing the embeddings
                (default: "dummy": random embeddings; will be passed to TransformerWordEmbeddings)
        - min_freq [int]: how often the word has to occur at least for an evolving embedding to be computed for it (default: 100)
        - n_tokens [int]: for how many tokens at most to compute the evolving embeddings (default: 10000)

    Returns:
        - snapshot_emb [dict: {str: SimplePretrainedEmbeddings}]: embedding snapshots for all snapshot dates. Please note that the dates used
                as the keys in the dictionary correspond to the actual dates where the snapshot was taken, which might differ from the snapshot date
                passed in `snapshots`, e.g., for a snapshot date '2020-02-31', the actual date of the snapshot will (most likely) be '2020-20-28'.
    """
    # transformer model to generate the local embeddings
    if local_emb_name.lower() == "dummy":
        local_emb = DummyEmbeddings(50, "testemb")  # for some quick testing only
    elif local_emb_name.lower() == "bert":  # shortcuts
        local_emb = TransformerWordEmbeddings("bert-base-uncased", layers="all", layer_mean=True, subtoken_pooling="mean", fine_tune=False)
    elif local_emb_name.lower() == "roberta":
        local_emb = TransformerWordEmbeddings("roberta-base", layers="all", layer_mean=True, subtoken_pooling="mean", fine_tune=False)
    else:
        local_emb = TransformerWordEmbeddings(local_emb_name, layers="all", layer_mean=True, subtoken_pooling="mean", fine_tune=False)
    # pass sentences directly to generate input model from all texts
    # so we know which words are of interest and their count (to set alphas manually afterwards)
    emb = EvolvingEmbeddings(local_emb, sentences, alpha=None, min_freq=min_freq, n_tokens=n_tokens, update_index=False)
    print("Number of words in the vocabulary (for which we'll learn an embedding):", emb.input_model.n_tokens)
    # create counts dict based on token_counts/interval
    counts_dict = {t: emb.input_model.token_counts[t]/(1.*len(snapshots)) for t in emb.input_model.index2token}
    # manually create max_counts array with entries for individual words
    emb._set_max_count(counts_dict)
    # compute evolving embeddings and take snapshot at the end of each split
    snapshot_emb = {}
    current_snap = 0
    for i, s in enumerate(sentences):
        if not i % 100:
            print(f"Processing sentence {i:8}/{len(sentences)}", end="\r")
        # check if we need to take the snapshot
        if dates[i] > snapshots[current_snap]:
            actual_snap = dates[i-1]  # the given snapshot might be at the 31st but the month could have only 30 days
            temp = emb.as_pretrained()
            # save as a model without flair dependency
            snapshot_emb[actual_snap] = SimplePretrainedEmbeddings(temp.embeddings, temp.input_model)
            # set OOV embedding to zeros
            snapshot_emb[actual_snap].embeddings[-1] = np.zeros(snapshot_emb[actual_snap].embeddings.shape[1])
            current_snap += 1
        # update embeddings with sentence
        emb.update_evolving_embeddings(s)
    # possibly take last snapshot
    if current_snap < len(snapshots):
        actual_snap = dates[i]
        temp = emb.as_pretrained()
        snapshot_emb[actual_snap] = SimplePretrainedEmbeddings(temp.embeddings, temp.input_model)
        snapshot_emb[actual_snap].embeddings[-1] = np.zeros(snapshot_emb[actual_snap].embeddings.shape[1])
    print(f"Processing sentence {len(sentences):8}/{len(sentences)}...done!")
    # reduce file size by ensuring dtype of numpy arrays is float32
    for s in snapshot_emb:
        snapshot_emb[s].embeddings = np.array(snapshot_emb[s].embeddings, dtype=np.float32)
    return snapshot_emb


def most_changed_tokens(snapshot_emb, ignore_zeros=True):
    """
    Check which tokens have changed the most over time.

    Inputs:
        - snapshot_emb [dict: {str: SimplePretrainedEmbeddings}]: embedding snapshots for all snapshot dates (as computed with compute_emb_snapshots)
        - ignore_zeros [bool]: whether to ignore tokens that only started occurring after the first snapshot, i.e., trivial changes (default: True)

    Returns:
        - [list: (str, simscore)]: list of tokens and associated score with most changed tokens first
    """
    # find the tokens whos embedding has changed the most over the whole time period
    snapshots = sorted(snapshot_emb)
    token_sim = []
    for t in snapshot_emb[snapshots[0]].input_model.index2token:
        # ignore the zero embeddings
        if ignore_zeros:
            token_emb = np.vstack([snapshot_emb[s][t] for s in snapshots if np.any(snapshot_emb[s][t] != 0)])
        else:
            token_emb = np.vstack([snapshot_emb[s][t] for s in snapshots])
        # overall sim = min/mean of upper triangular similarity values
        # -> take into account similarity of all emb to one another at all time points
        if token_emb.shape[0] > 1:
            sim = cosine_similarity(token_emb)
            if ignore_zeros:
                token_sim.append(sim[np.triu_indices(sim.shape[0], k=1)].min())
            else:
                token_sim.append(sim[np.triu_indices(sim.shape[0], k=1)].mean())
        else:
            token_sim.append(1)
    # sort index from smallest to largest - the more different the word, the smaller the sim
    token_idx = np.argsort(token_sim)
    tokens = [(snapshot_emb[snapshots[0]].input_model.index2token[i], token_sim[i]) for i in token_idx]
    return [t for t in tokens if t[0].isalnum()]


def analyze_emb_over_time(snapshot_emb, token, k=5, savefigs=""):
    """
    Generate time lines and PCA plots for the given token.

    Inputs:
        - snapshot_emb [dict: {str: SimplePretrainedEmbeddings}]: embedding snapshots for all snapshot dates (as computed with compute_emb_snapshots)
        - tokens [str]: the token for which the plots should be generated
        - k [int]: how many nearest neighbors should be included in the plots at most (default: 5)
        - savefigs [str]: if given, save matplotlib figures at "savefigs_token_...pdf"

    Returns:
        - fig_time, fig_pca: the two plotly figures (or None, None if token was not in embedding snapshots)
    """
    token = token.lower()
    snapshots = sorted(snapshot_emb)
    # check if token is known
    if token not in snapshot_emb[snapshots[0]]:
        return None, None
    # get the two snapshots where the embeddings of the token are the most different
    snapshots_nonz = [s for s in snapshots if np.any(snapshot_emb[s][token] != 0)]
    if len(snapshots_nonz) > 1:
        token_emb = np.vstack([snapshot_emb[s][token] for s in snapshots_nonz])
        sim = cosine_similarity(token_emb)
        rowidx, colidx = np.triu_indices(sim.shape[0], k=1)
        minidx = sim[rowidx, colidx].argmin()
        first, last = snapshots_nonz[rowidx[minidx]], snapshots_nonz[colidx[minidx]]
    else:
        first, last = snapshots[0], snapshots_nonz[0]

    # get the corresponding nearest neighbors
    nn_first = snapshot_emb[first].get_nneighbors(token, k, include_simscore=False)
    nn_last = snapshot_emb[last].get_nneighbors(token, k, include_simscore=False)

    # get colors for plots later
    colors = {}
    colors[token] = (0., 0., 0., 1.)
    colors[f"{token} ({first})"] = (0., 0., 0., 1.)
    colors[f"{token} ({last})"] = (0., 0., 0., 1.)
    cmap = plt.get_cmap("RdBu")
    for i, t in enumerate(nn_first):
        colors[t] = cmap(0.4*(i/(k-1)))
    for i, t in enumerate(nn_last):
        if t not in colors:
            colors[t] = cmap(1-0.4*(i/(k-1)))
    # plotly colors (careful - wants css colors)
    color_plotly = {t: f"rgb{tuple(k*255 for k in v[:3])}" for t, v in colors.items()}
    # make sure nn_last only contains tokens not in nn_first
    nn_last = [t for t in nn_last if t not in nn_first]

    # create embedding matrices per token over time
    token_emb = {}
    for t in [token] + nn_first + nn_last:
        token_emb[t] = np.vstack([snapshot_emb[s][t] for s in snapshots])

    # compute similarity of each nn to the token
    sim_scores = {}
    for t in nn_first + nn_last:
        sim_scores[t] = np.diag(cosine_similarity(token_emb[token], token_emb[t]))
    # similarity of token itself to first and last embedding
    sim_scores[f"{token} ({first})"] = cosine_similarity(token_emb[token], token_emb[token][[snapshots.index(first)]]).flatten()
    sim_scores[f"{token} ({last})"] = cosine_similarity(token_emb[token], token_emb[token][[snapshots.index(last)]]).flatten()

    # plot evolution of similarity scores over time
    plot_tokens = [f"{token} ({first})", f"{token} ({last})"] + nn_first + nn_last

    if savefigs:
        snapshot_dates = list(range(len(snapshots)))
        plt.figure(figsize=(8, 5))
        for t in plot_tokens:
            plt.plot(snapshot_dates, sim_scores[t], "--" if t == f"{token} ({first})" else "-", color=colors[t], label=t)
        l = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=14)
        plt.xticks(snapshot_dates, snapshots, rotation=90 if len(snapshots) > 5 else 0)
        plt.ylabel("cosine similarity")
        # plt.title(token)
        plt.savefig(f"{savefigs}_{token}_{snapshots[0]}_{snapshots[-1]}_time.pdf", dpi=300, bbox_inches="tight", bbox_extra_artists=[l])

    # interactive timelines with plotly
    df_temp = pd.DataFrame({
        "snapshot date": [datetime.strptime(s, '%Y-%m-%d') for t in plot_tokens for s in snapshots],
        "cosine similarity": np.array([sim_scores[t] for t in plot_tokens]).flatten(),
        "token": [t for t in plot_tokens for s in snapshots],
        "line": ["dash" if t == f"{token} ({first})" else "solid" for t in plot_tokens for s in snapshots]
    })
    fig_time = px.line(df_temp, x="snapshot date", y="cosine similarity",
                       color="token", color_discrete_map=color_plotly, hover_name="token",
                       line_dash="line", line_dash_map='identity', height=500)

    # plot 2D PCA vis of embeddings
    full_embedding_mat = []
    labels = []
    color_keys = []
    size = []
    for t in [token] + nn_first + nn_last:
        full_embedding_mat.append(token_emb[t])
        labels.extend([f"{t} ({s})" for s in snapshots])
        color_keys.extend(len(snapshots)*[t])
        size.extend(list(range(1, len(snapshots) + 1)))
    full_embedding_mat = np.vstack(full_embedding_mat)
    X_kpca = KernelPCA(n_components=2, kernel="cosine").fit_transform(full_embedding_mat)

    # with matplotlib
    if savefigs:
        plt.figure(figsize=(6, 6))
        plt.scatter(x=X_kpca[:, 0], y=X_kpca[:, 1], s=10*np.array(size), c=[colors[t] for t in color_keys], alpha=0.6)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        # plt.title(token)
        plt.savefig(f"{savefigs}_{token}_{snapshots[0]}_{snapshots[-1]}_pca.pdf", dpi=300, bbox_inches="tight")

    # interactive with plotly
    fig_pca = px.scatter(x=X_kpca[:, 0], y=X_kpca[:, 1], color=color_keys, size=np.sqrt(size), color_discrete_map=color_plotly, hover_name=labels, height=500, width=650)
    fig_pca.update_traces(hovertemplate='%{hovertext}')  # only show our text, no additional info
    return fig_time, fig_pca
