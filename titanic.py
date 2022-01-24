"""
author: Scott F

generates a python decision-tree classifier
at out\titanic_classifier.py

TODO
- clean data, keep only binary attributes
- expand find_split_attr, build_tree, and split_data to more
  than just binary attributes
"""
import csv
import math
from re import T
import pandas as pd
import numpy as np


def main():
    # set build parameters
    training_file = r'data/train.csv'    
    validation_file = r'data/test.csv'
    out_file = 'out/titanic_classifier.py'
    target_attr = 'Survived'
    
    # pull csv data
    with open(training_file) as csv_file:
        data = pd.read_csv(csv_file)

    # find column associated w/ target variable
    target_col = data.columns.get_loc(target_attr)
    data = clean_titanic_data(data)

    # build decision tree string
    tree = build_tree(data, target_attr)

    # write classifier file
    emit_prologue(out_file, validation_file)
    emit_body(out_file, tree)
    emit_epilogue(out_file)


def clean_titanic_data(data):
    """
    converts all of the strings in a 2d dataset (except the column
    headers) to floats

    :param data: 2d list of values, first row is column headers

    :returns: cleaned data
    """
    return data[["PassengerId", "Survived", "Sex"]].replace("male", 1).replace("female", 0)


def emit_prologue(out_file, validation_file):
    """
    Writes header information to the output program, including file
    docstrings and imports

    :param out_file: the filename for the new program
    :param validation_file: the filename for the validation data set
    """
    # open and overwrite the new classifier file
    fd = open(out_file, 'w')

    # write header information and open validation file # TODO update
    fd.write(
        "\"\"\"\n" \
        "author: Scott F\n" \
        "\"\"\"\n" \
        "import pandas as pd\n" \
        "\n" \
        "def main():\n" \
        "    # open validation data\n" \
        "    with open(\"{}\") as fd:\n".format(validation_file) + \
        "        data = pd.DataFrame(fd)\n"
        "\n" \
    )

    # close new classifier file
    fd.close()


def emit_body(out_file, tree_str):
    """
    Writes the classification info to the new classifier file

    :param out_file: the filename for the new program
    :param col_id: the int id of the column that the classifier is using
    """
    # append to the header of the program
    fd = open(out_file, 'a')

    # write classification info derived from "find_split_attr()"
    fd.write(
        "    # iterates through each row and classifies the target\n" \
        "    # classifications are printed to standard out\n" \
        "    for idx, row in data.iterrows():\n" + \
        tree_str + \
        "\n"
    )

    # close new classifier file
    fd.close()


def emit_epilogue(out_file):
    """
    Writes the closing call the the main method to the classifier file

    :param out_file: the filename for the new program
    """
    # append to header + body
    fd = open(out_file, 'a')

    # write call to main
    fd.write(
        "\n" \
        "if __name__ == \"__main__\":\n" \
        "    main()\n" 
    )

    # close completed classifier file
    fd.close()


def find_split_attr(data, target_col):
    """
    Find the best attribute to use for a classifier for the given target
    variable

    :param data: 2d array of data values
    :param target_col: the column number of the target variable

    :returns: the column id of the best attribute to split on and its entropy
    """
    # initialize "best" values
    best_col_id = ''
    min_bhatt = float('inf')

    # try the splitting with each column
    for col in data.columns:
        # Node 1: col == 0
        # Class 1: target_col == 0
        # Class 2: target_col == 1
        n1_c1 = 0
        n1_c2 = 0

        # Node 2: col == 0
        # Class 1: target_col == 0
        # Class 2: target_col == 1
        n2_c1 = 0
        n2_c2 = 0
        
        # filter each data point into the correct node and class
        for _, row in data.iterrows():

            if row[col] == 0:
                if row[target_col] == 0:
                    n1_c1 += 1
                elif row[target_col] == 1:
                    n1_c2 += 1
            elif row[col] == 1:
                if row[target_col] == 0:
                    n2_c1 += 1
                elif row[target_col]:
                    n2_c2 += 1

        # calculate bhatt. coeff as metric for splitting
        try: 
            n1_p1 = n1_c1/(n1_c1+n1_c2) # probability of Node 1 Class 1
        except ZeroDivisionError:
            n1_p1 = 0
        try:
            n1_p2 = n1_c2/(n1_c1+n1_c2) # probability of Node 1 Class 2
        except ZeroDivisionError:
            n1_p2 = 0
        try:
            n2_p1 = n2_c1/(n2_c1+n2_c2) # probability of Node 2 Class 1
        except ZeroDivisionError:
            n2_p1 = 0
        try: 
            n2_p2 = n2_c2/(n2_c1+n2_c2) # probability of Node 2 Class 2
        except ZeroDivisionError:
            n2_p2 = 0
        
        bhatt = math.sqrt(n1_p1*n2_p1) + math.sqrt(n1_p2*n2_p2)

        # ignore non-binary attributes
        if bhatt == 0:
            continue

        # update best values
        if bhatt < min_bhatt:
            best_col_id = col
            min_bhatt = bhatt

    return best_col_id


def build_tree(data, target, depth=0, l_r=0):
    """
    recursively finds the best attributes to split on and builds a decision
    tree until the depth is 3 or a node is 95% pure

    :param data: 2d array of data values
    :param depth: current recursive depth of the tree
    :param target: target column
    :param l_r: left or right, which side of the tree is it on, used when hit leaf node
    :returns: a string containing the if-else structure of the decision tree
    """
    # base case
    if depth == 1 or node_purity(data, target) >= .95: # 2 starting from 0
        return ("    " * (depth+2)) + "print({})\n".format(l_r)

    else:
        attr = find_split_attr(data, target)
        l_data, r_data = split_data(data, attr)

        return  ("    " * (depth+2)) + "# attribute\n".format(attr) +\
                ("    " * (depth+2)) + "if int(row[\'{}\']) == 1:\n".format(attr) + \
                build_tree(l_data, target, depth+1, 1) + \
                ("    " * (depth+2)) + "else:\n" + \
                build_tree(r_data, target, depth+1, 0)


def node_purity(data, target):
    """
    Checks for homogeneity within a node

    :param data: 2d array of data values
    :param target: attribute we are considering for purity  
    """
    zeros = data[target].value_counts()[0]
    ones  = data[target].value_counts()[1]

    return max([zeros/(zeros+ones), ones/(zeros+ones)])


def split_data(data, attr):
    """
    spits a dataset at a given attribute into two new sets
    
    :param data: pandas dataframe
    :param attr: attribute to split on
    :returns: l_data and r_data where l_data is all rows were attr == 0
              and r_data is all rows where attr == 1
    """
    l_data = [] # data left of the threshold (attr == 0)
    r_data = [] # data right of the threshold (attr == 1)
    print(attr)
    # copy each row
    for _, row in data.iterrows():
        if row[attr] == 0:
            l_data.append(row)
        else:
            r_data.append(row)

    return pd.DataFrame(l_data, columns=data.columns), \
           pd.DataFrame(r_data, columns=data.columns)


if __name__ == "__main__":
    main()
