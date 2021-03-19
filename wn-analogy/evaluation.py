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

def suggestions_to_csv(suggestions, users, outfile, sample_size, relations, form_lemma_dict):
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
    wrsample = data_df[["wordA","relation"]].drop_duplicates()
    wrsample = wrsample.groupby("relation").sample(sample_size)
    logger.debug(f"sample table for relation\n {wrsample}")
    
    wrsample["aux"] = wrsample["wordA"] + wrsample["relation"]
    data_df["aux"] =  data_df["wordA"] + data_df["relation"]
    data_df = data_df[data_df["aux"].isin(wrsample["aux"])]

    # ordering colums
    ordered  = ["wordA","lemmaA","posA","wordB","lemmaB","posB"]
    ordered += ["relation","hit","valid"] + methods + users
    data_df = data_df.reindex(columns=ordered)

    # ordering by relations and replacing examples
    data_df = data_df.sort_values(by=["relation","wordA","wordB"])
    data_df["relation"] = data_df["relation"].apply(_get_example)

    # replaces bool to integer
    data_df["hit"] = data_df["hit"].apply(int)
    data_df["valid"] = data_df["valid"].apply(int)

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
                
                if form in form_lemma_dict.keys():
                    if pos in form_lemma_dict[form].keys():
                        lemmas = form_lemma_dict[form][pos].split("/")
                        lemmas += [lemma] if lemma not in lemmas else []
                        form_lemma_dict[form][pos] = "/".join(lemmas)
                    else:
                        form_lemma_dict[form][pos] = lemma
                else:
                    form_lemma_dict[form] = {pos:lemma}

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

    domain = _relations[relation]["domain"]
    _range = _relations[relation]["range"]

    # if valid relation and both lemmas exist and are different
    if lemmaA != lemmaB and lemmaA and lemmaB and typeA in domain and typeB in _range:
        return True,posA,lemmaA,posB,lemmaB

    # case typeA is CoreConcept checks all domain
    if typeA == "CoreConcept":
        #search a valid type for the relation
        for typeA in domain:
            posA,lemmaA = _get_lemma(form_lemma_dict, wordA, typeA)
            if lemmaA != lemmaB and lemmaA and lemmaB and typeB in _range:
                return True,posA,lemmaA,posB,lemmaB
    
    # not valid relation if
    # lemas are the same
    # one lemma doesnt exist
    # pos not valid for relation
    return False,posA,lemmaA,posB,lemmaB
     

def _get_lemma(form_lemma_dict, word, type):
    try:
        if type == "NounSynset": return "N",form_lemma_dict[word]["N"]
        if type == "VerbSynset": return "V",form_lemma_dict[word]["V"]
        if type == "AdverbSynset": return "ADV",form_lemma_dict[word]["ADV"]
        if type == "AdjectiveSynset": return "A",form_lemma_dict[word]["A"]
    except:
        return None,None

def _get_example(relation):
    for key, value in _relations.items():
        if relation.find(key) >= 0:
            return value["example"]
    return key


_relations = {
    'attribute': {
        'domain': ['NounSynset'],
        'range': ['AdjectiveSynset'],
        'example': 'attribute X-N Y-A : crânio -> duro'},
    'causes': {
        'domain': ['VerbSynset'],
        'range': ['VerbSynset'],
        'example': 'causes X-V Y-V: derrubar -> cair'},
    'partHolonymOf': {
        'domain': ['NounSynset'], 
        'range': ['NounSynset'],
        'example':'partHolonymOf A-N B-N : ancora -> barco'},
    'antonymOf': {
        'domain': ["AdjectiveSynset"],
        'range': ["AdjectiveSynset"],
        'example': 'antonymOf X-? Y-? : casado <-> solteiro'},
    'entails': {
        'domain': ['VerbSynset'],
        'range': ['VerbSynset'],
        'example': 'entails X-V Y-V : cantar -> soar'},
    'memberHolonymOf': {
        'domain': ['NounSynset'], 
        'range': ['NounSynset'],
        'example': 'memberHolonymOf X-N Y-N : auditor -> audiência'},
    'memberMeronymOf': {
        'domain': ['NounSynset'], 
        'range': ['NounSynset'],
        'example':'memberMeronymOf X-N Y-N : audiencia -> auditor'},
    'partMeronymOf': {
        'domain': ['NounSynset'], 
        'range': ['NounSynset'],
        'example':'partMeronymOf X-N Y-N : ônibus -> janela'},
    'substanceHolonymOf': {
        'domain': ['NounSynset'], 
        'range': ['NounSynset'],
        'example':'substanceHolonymOf X-N Y-N : osso -> chifre'},
    'substanceMeronymOf': {
        'domain': ['NounSynset'], 
        'range': ['NounSynset'],
        'example':'substanceHolonymOf X-N Y-N : chifre -> osso'},
    'agent': {
        'domain': ['VerbSynset'], 
        'range': ['NounSynset'],
        'example':'substanceHolonymOf X-V Y-N : cantar -> cantor'},
    'byMeansOf': {
        'domain': ['VerbSynset'], 
        'range': ['NounSynset'],
        'example':'substanceHolonymOf X-V Y-N : acalmar -> calmante'}
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