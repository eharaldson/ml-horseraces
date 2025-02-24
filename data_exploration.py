import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv('ProcessedDataset.csv', index_col=0)

    for column in df.columns:
        print(column)
        plt.hist(df[column], bins=20)
        plt.title(column)
        plt.show()