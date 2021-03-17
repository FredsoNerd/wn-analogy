# -*- coding: utf-8 -*-

import json
import pandas as pd
import argparse


def _parse(args):
    users = args.u
    outfile = args.o
    filenames = args.f  
    
    suggestions = []
    for filename in filenames:
        with open(filename) as file:
            suggestions += json.load(file)
            # suggestions.append(json.load(file)[0])
    
    # cals main function
    suggestions_to_csv(suggestions, users, outfile)

def suggestions_to_csv(suggestions, users, outfile):
    """
    formats a csv containing suggestions for each method for
    human evaluation. The output should be of the form:
    """
    
    dataset = []

    for s in suggestions:
        details = s["details"]
        methods = s["experiment_setup"]["method"]

        # formats the data
        dataset += [{
            "wordA": detail["b"],
            "wordB": prediction["answer"],
            "method": methods
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
    data_df = data_df.groupby(["wordA","wordB"]).agg(aggregation)
    data_df = data_df.reset_index()

    # adds users to vote
    for user in users:  data_df[user] = 0
    print(data_df)

    data_df.to_csv(outfile)
    

# sets parser and interface function
parser = argparse.ArgumentParser()

# sets the user options
parser.add_argument("-f", help="dataset files", nargs="+")
parser.add_argument("-u", help="users to vote", nargs="+")
parser.add_argument("-o", help="output filename (default: output.csv)", default="output.csv")

# cals the parser
_parse(parser.parse_args())