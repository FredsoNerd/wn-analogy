# -*- coding: utf-8 -*-

import os
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
    
    morphobrpath = args.m
    morphobrfiles = _find_files(morphobrpath, ".dict")

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
        method = s["experiment_setup"]["method"]
        relation = s["experiment_setup"]["subcategory"].split(".")[0]
        logger.debug(f"structuring details from {method}-{relation}")

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
                valid,posA,lemmaA,posB,lemmaB = _validate_relation(relation,wordA,wordB,form_lemma_dict)
                # ## ignores if not in the morphobr
                # if form_lemma_dict and not valid:
                #     continue

                data = dict()
                data["hit"] = ishit
                data["posA"] = posA
                data["wordA"] = wordA
                data["posB"] = posB
                data["wordB"] = wordB
                data["valid"] = valid
                data["lemmaA"] = lemmaA
                data["lemmaB"] = lemmaB
                data["method"] = method
                data["relation"] = relation
                dataset.append(data) 
         
    # formats data as dataframe and add dummies
    data_df = pd.DataFrame(dataset)
    methods = list(data_df["method"].drop_duplicates())
    dummies = pd.get_dummies(data_df["method"])
    data_df = data_df.drop("method", axis=1)
    data_df = data_df.join(dummies)
    # group and aggregate
    aggregation = {col:"sum" for col in dummies.columns}
    aggregation["hit"] = "min"
    aggregation["valid"] = "min"
    aggregation["posA"] = "min"
    aggregation["lemmaA"] = "min"
    aggregation["posB"] = "min"
    aggregation["lemmaB"] = "min"
    data_df = data_df.groupby(["wordA","wordB","relation"]).agg(aggregation)
    data_df = data_df.reset_index()

    # choses a sample for each relation
    word_rel_sample = data_df[["wordA","relation"]].drop_duplicates().sample(sample)
    data_df = data_df[data_df.wordA.isin(word_rel_sample.wordA) & data_df.relation.isin(word_rel_sample.relation)]

    # ordering colums
    ordered  = ["wordA","lemmaA","posA","wordB","lemmaB","posB"]
    ordered += ["relation","hit","valid"] + methods + users
    data_df = data_df.reindex(columns=ordered)

    # sets users votes as
    data_df.loc[data_df.valid == False, users] = 0    

    # saves and shows output
    logger.info(f"saving output to file {outfile}")
    logger.info(f"output table \n {data_df}")
    data_df.to_csv(outfile, index=False)


def _read_from_dicts(filenames):
    form_lemma_dict = dict()
    # updates the lemmas dictionary
    for filename in filenames:
        logger.debug(f"reading forms/lemmas from file {filename}")
        with open(filename) as dictfile:
            for dictline in dictfile:
                # formats each line 
                form,metadata = dictline.split()
                lemma = metadata.split("+")[0]
                # pos in one of N,V,A,ADV
                pos = metadata.split("+")[1].split(".")[0]
                
                try: form_lemma_dict[form][pos] = lemma
                except: form_lemma_dict[form] = {pos:lemma}

    return form_lemma_dict

def _find_files(filepaths, extension):
    found = []
    for filepath in filepaths:
        if os.path.isdir(filepath):
            for root, _, files in os.walk(filepath):
                found += [os.path.join(root,f) for f in files if f.endswith(extension)]
        elif os.path.isfile(filepath) and filepath.endswith(extension):
            found.append(filepath)
    
    return found

def _validate_relation(relation, wordA, wordB, form_lemma_dict):
    relation,typeA,typeB = relation.split("-")
    relation = relation.split("_")[-1]

    posA,lemmaA = _get_lemma(form_lemma_dict, wordB, typeB)
    posB,lemmaB = _get_lemma(form_lemma_dict, wordB, typeB)

    domain = _domain_range_map[relation]["domain"]
    _range = _domain_range_map[relation]["range"]

    # if valid relation and both lemmas exist
    if lemmaA and lemmaB and typeA in domain and typeB in _range:
        return True, lemmaA, lemmaB

    # case typeA is CoreConcept checks all domain
    if typeA == "CoreConcept":
        #search a valid type for the relation
        for typeA in domain:
            posA,lemmaA = _get_lemma(form_lemma_dict, wordA, typeA)
            if lemmaA and lemmaB and typeB in _range:
                return True,posA,lemmaA,posB,lemmaB
    
    return False,posA,lemmaA,posB,lemmaB
     

def _get_lemma(form_lemma_dict, word, type):
    try:
        if type == "NounSynset": return "N",form_lemma_dict[word]["N"]
        if type == "VerbSynset": return "V",form_lemma_dict[word]["V"]
        if type == "AdverbSynset": return "ADV",form_lemma_dict[word]["ADV"]
        if type == "AdjectiveSynset": return "A",form_lemma_dict[word]["A"]
    except:
        return None,None


_domain_range_map = {
    'attribute': {
        'domain': ['NounSynset'],
        'range': ['AdjectiveSynset']},
    'causes': {
        'domain': ['VerbSynset'],
        'range': ['VerbSynset']},
    'entails': {
        'domain': ['VerbSynset'],
        'range': ['VerbSynset']},
    'memberHolonymOf': {
        'domain': ['NounSynset'], 
        'range': ['NounSynset']},
    'memberMeronymOf': {
        'domain': ['NounSynset'], 
        'range': ['NounSynset']},
    'partHolonymOf': {
        'domain': ['NounSynset'], 
        'range': ['NounSynset']},
    'partMeronymOf': {
        'domain': ['NounSynset'], 
        'range': ['NounSynset']},
    'substanceHolonymOf': {
        'domain': ['NounSynset'], 
        'range': ['NounSynset']},
    'substanceMeronymOf': {
        'domain': ['NounSynset'], 
        'range': ['NounSynset']},
    'agent': {
        'domain': ['VerbSynset'], 
        'range': ['NounSynset']},
    # 'antonymOf': {
    #     'domain': [], 
    #     'range': []},
    'byMeansOf': {
        'domain': ['VerbSynset'], 
        'range': ['NounSynset']}
    }


# sets parser and interface function
parser = argparse.ArgumentParser()

# sets the user options
parser.add_argument("-f", help="dataset files", nargs="+")
parser.add_argument("-u", help="users to vote", nargs="+")
parser.add_argument("-n", help="words sample size (default value: 10)", type=int, default=10)
parser.add_argument("-m", help="dict filepaths (no filters if none)", nargs="*", default=[])
parser.add_argument("-r", help="relations to filter (no filters if none)", nargs="*", default=[])
parser.add_argument("-o", help="output filename (default value: output.csv)", default="output.csv")

parser.add_argument("-v", help="increase verbosity (example: -vv for debugging)", action="count", default=0)

# cals the parser
_parse(parser.parse_args())