# -*- coding: utf-8 -*-

import json
import pandas as pd

import argparse
import logging
logger = logging.getLogger("evaluation")


def _parse(args):
    users = args.u
    sample = args.n
    outfile = args.o
    filenames = args.f
    relations = args.r
    verbosity = args.v
    morphobrfiles = args.m

    # sets verbosity level
    logging.basicConfig(level= 30-10*verbosity)

    # mounts form lemma dict
    logger.info("mounting form lemma dict from MorphoBR")
    form_lemma_dict = _read_from_dicts(morphobrfiles)
    
    # collects json outputs
    logger.info("collecting dataset from files")
    suggestions = []
    for filename in filenames:
        logger.debug(f"collecting json from {filename}")
        with open(filename) as file:
            suggestions += json.load(file)
    
    # cals main function
    suggestions_to_csv(suggestions, users, outfile, sample, relations, form_lemma_dict)

def suggestions_to_csv(suggestions, users, outfile, sample, relations, form_lemma_dict):
    """
    formats a csv containing suggestions for each method for
    human evaluation, given the users to eval.
    """
    
    dataset = []

    for s in suggestions:
        details = s["details"]
        methods = s["experiment_setup"]["method"]
        relation = s["experiment_setup"]["subcategory"].split(".")[0]
        logger.debug(f"structuring details from {methods}-{relation}")

        # filters by relations, if given
        if relations and relation not in relations:
            continue

        # formats the data
        for detail in details:
            wordA = detail["b"]
            for prediction in detail["predictions"]:
                ishit = prediction["hit"]
                wordB = prediction["answer"]
                
                # filters by form-lemma, if given
                lemmaA = _get_lemma(form_lemma_dict,wordA)
                lemmaB = _get_lemma(form_lemma_dict,wordB)
                ## ignores if not in the morphobr
                if form_lemma_dict and (not lemmaA or not lemmaB):
                    continue
                # if form_lemma_dict and lemmaA == lemmaB:
                #     logger.debug(f"got the same lemma {wordA} {wordB}, on relation {relation}.")
                #     continue

                data = dict()
                data["hit"] = ishit
                data["wordA"] = wordA
                data["wordB"] = wordB
                data["method"] = methods
                data["relation"] = relation
                dataset.append(data) 
         
    # formats data as dataframe and add dummies
    data_df = pd.DataFrame(dataset)
    dummies = pd.get_dummies(data_df["method"])
    data_df = data_df.drop("method", axis=1)
    data_df = data_df.join(dummies)
    # group and aggregate
    aggregation = {col:"sum" for col in dummies.columns}
    data_df = data_df.groupby(["wordA","wordB", "relation", "hit"]).agg(aggregation)
    data_df = data_df.reset_index()

    # adds users to vote
    for user in users: data_df[user] = 0

    # shoses a sample
    words = data_df["wordA"].drop_duplicates().sample(sample)
    data_df = data_df[data_df.wordA.isin(words)].reset_index()

    # saves and shows output
    logger.info(f"saving output to file {outfile}")
    logger.info(f"output table \n {data_df}")
    data_df.to_csv(outfile)
    
def _get_lemma(form_lemma_dict, word):
    try: return form_lemma_dict[word]
    except: return None

def _read_from_dicts(filenames):
    form_lemma_dict = dict()
    # updates the lemmas dictionary
    for filename in filenames:
        logger.debug(f"reading forms/lemmas from file {filename}")
        with open(filename) as dictfile:
            for dictline in dictfile:
                # formats each line 
                form,lemma = dictline.split("+")[0].split()
                form_lemma_dict[form] = lemma

    return form_lemma_dict


# sets parser and interface function
parser = argparse.ArgumentParser()

# sets the user options
parser.add_argument("-f", help="dataset files", nargs="+")
parser.add_argument("-u", help="users to vote", nargs="+")
parser.add_argument("-n", help="words sample size (default value: 10)", type=int, default=10)
parser.add_argument("-m", help="MorphoBR to filter (no filters if none)", nargs="*", default=[])
parser.add_argument("-r", help="relations to filter (no filters if none)", nargs="*", default=[])
parser.add_argument("-o", help="output filename (default value: output.csv)", default="output.csv")

parser.add_argument("-v", help="increase verbosity (example: -vv for debugging)", action="count", default=0)

# cals the parser
_parse(parser.parse_args())