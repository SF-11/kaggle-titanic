"""
author: Scott F
"""
import pandas as pd

def main():
    # open validation data
    with open("data/test.csv", 'r') as fd:
        data = pd.read_csv(fd)

    # iterates through each row and classifies the target
    # classifications are printed to standard out
    for idx, row in data.iterrows():
        # attribute
        
        if row['Sex'] == 'male':
            print(f"{row['PassengerId']},{0}")
        else:
            print(f"{row['PassengerId']},{1}")
            

if __name__ == "__main__":
    main()
