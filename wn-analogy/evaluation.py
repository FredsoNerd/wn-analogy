# -*- coding: utf-8 -*-

import json
import pandas as pd
import argparse


def _parse(args):
    users = args.u
    sample = args.n
    outfile = args.o
    filenames = args.f
    
    suggestions = []
    for filename in filenames:
        with open(filename) as file:
            suggestions += json.load(file)
            # suggestions.append(json.load(file)[0])
    
    # cals main function
    suggestions_to_csv(suggestions, users, outfile, sample)

def suggestions_to_csv(suggestions, users, outfile, sample):
    """
    formats a csv containing suggestions for each method for
    human evaluation. The output should be of the form:
    """
    
    dataset = []

    for s in suggestions:
        details = s["details"]
        methods = s["experiment_setup"]["method"]
        relation = s["experiment_setup"]["subcategory"].split(".")[0]

        # formats the data
        dataset += [{
            "wordA": detail["b"],
            "wordB": prediction["answer"],
            "method": methods,
            "relation": relation,
            "hit": prediction["hit"]
            }
            for detail in details
            for prediction in detail["predictions"]]
         
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
    print(data_df)
    data_df.to_csv(outfile)
    

# sets parser and interface function
parser = argparse.ArgumentParser()

# sets the user options
parser.add_argument("-f", help="dataset files", nargs="+")
parser.add_argument("-u", help="users to vote", nargs="+")
parser.add_argument("-o", help="output filename (default: output.csv)", default="output.csv")
parser.add_argument("-n", help="wordsA sample size (default: 100)", type=int, default=100)

# cals the parser
_parse(parser.parse_args())

# 0 ok
# 1 ok
# 2 fazer
# 3 ok nao fazer
# 5 fazer
# 6 fazer
# 7 não
# 8 não