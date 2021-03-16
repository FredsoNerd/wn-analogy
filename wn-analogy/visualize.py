# -*- coding: utf-8 -*-

from vecto.data import Dataset
from vecto.embeddings import load_from_dir 
from vecto.benchmarks.analogy.io import get_pairs

from numpy import vstack
from numpy.linalg import norm

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import argparse
# import logging
# logger = logging.getLogger(__name__)


def _parse(args):
    embs = load_from_dir(args.embs)
    dataset = Dataset(args.data)
    visulize_relations(embs, dataset)


def visulize_relations(embs, dataset):
    # format pairs (wordA, [wordB])
    pairs = []
    for filename in dataset.file_iterator():
        pairs += get_pairs(filename)

    offsets = []
    # all pais of related words
    related_words = [(wa,wb) for (wa, wsb) in pairs for wb in wsb]
    for wa,wb in related_words:
        # not pair missing
        if embs.vocabulary.get_id(wa) < 0: continue
        if embs.vocabulary.get_id(wb) < 0: continue
        
        # normed vectors offset
        va = embs.get_vector(wa)
        vb = embs.get_vector(wb)
        # offsets.append(va - vb)
        offsets.append(va/norm(va) - vb/norm(vb))

    
    # prepare frame
    fig, ax = plt.subplots(1,2)
    fig.suptitle(dataset.path)
    
    # add mean to end of list
    offsets.append(vstack(offsets).mean(axis=0))

    # with t-sne
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(offsets.copy())
    
    ax[0].scatter(tsne_result[:,0],tsne_result[:,1])
    ax[0].scatter(tsne_result[-1,0],tsne_result[-1,1])
    ax[0].set_title("t-SNE Mean: {}".format(tsne_result[-1]))

    # with pca
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(offsets.copy())

    ax[1].scatter(pca_result[:,0], pca_result[:,1])
    ax[1].scatter(pca_result[-1,0], pca_result[-1,1])
    ax[1].set_title("PCA Mean: {}".format(pca_result[-1]))
    
    plt.show()


# sets parser and interface function
parser = argparse.ArgumentParser()

# sets the user options
parser.add_argument("embs", help="embeddings dir path")
parser.add_argument("data", help="dataset dir path")
# parser.add_argument("data", help="dataset files", nargs="+")

# cals the parser
_parse(parser.parse_args())