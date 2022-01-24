"""
author: Scott F
"""
import csv

def main():
    # open validation data
    with open("data/test.csv") as fd:
        data = list(csv.reader(fd))

    # iterates through each row and classifies the target
    # classifications are printed to standard out
    for i in range(1, len(data)):
        print(0)


if __name__ == "__main__":
    main()
