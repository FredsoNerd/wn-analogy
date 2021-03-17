# -*- coding: utf-8 -*-


from vecto.data import Dataset
from vecto.embeddings import load_from_dir 
from vecto.benchmarks.analogy.io import get_pairs
from vecto.benchmarks.analogy import Benchmark as Analogy

import json
import argparse
import logging
logger = logging.getLogger(__name__)


def _parse(args):
    embs = load_from_dir(args.embs)
    dataset = Dataset(args.data)
    # cals main function
    apply_method_analogy(embs, dataset)


def apply_method_analogy(embs, dataset):
    #Métodos a testar (SimilarToB é a baseline)
    metodos = ["SimilarToB", "3CosAvg", "LRCos"]
    for method in metodos:
        logger.info("Run analogy... {}".format(method))
        # run analogy
        analogy = Analogy(method=method)
        result = analogy.run(embs, dataset)
        # saves to files
        filename = "resultados_{}.json".format(method)
        logger.info("Save JSON... {}".format(filename))
        with open(filename, "w+") as outfile:
            json.dump(result, outfile, indent=4, ensure_ascii=False)


# sets parser and interface function
parser = argparse.ArgumentParser()

# sets the user options
parser.add_argument("embs", help="embeddings dir path")
parser.add_argument("data", help="dataset dir path")

# cals the parser
_parse(parser.parse_args())