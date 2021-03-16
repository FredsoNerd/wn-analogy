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
    apply_methods_train_test(embs, dataset)


def apply_methods_train_test(embs, dataset):
    pass

# sets parser and interface function
parser = argparse.ArgumentParser()

# sets the user options
parser.add_argument("embs", help="embeddings dir path")
parser.add_argument("train_data", help="train_dataset dir path")
parser.add_argument("test_data", help="test dataset dir path")

# cals the parser
_parse(parser.parse_args())