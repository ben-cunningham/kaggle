import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score

def handle_dates(df):
    pass

if __name__ == '__main__':
    test_df = pd.read_csv('../data/airbnb-new-user/train_users.csv')
    handle_dates(test_df)
