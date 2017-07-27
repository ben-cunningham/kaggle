import random
from dateutil.parser import parse

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score


def as_month(mo):
    if mo <= 3:
        return 1
    elif mo > 3 and mo <= 6:
        return 2
    elif mo > 6 and mo <= 9:
        return 3
    else:
        return 4

def handle_dates(df):
    days_to_book = []
    season = []
    for index, row in df.iterrows():
        if pd.isnull(row['date_first_booking']):
            days_to_book.append(-1)
            season.append(-1)
        else:
            days = parse(row['date_first_booking']) - parse(row['date_account_created'])
            days_to_book.append(abs(days.days))
            date = parse(row['date_first_booking'])
            season.append(as_month(date.month))
    df['days_to_book'] = days_to_book
    df['season'] = season
    
    df.drop('date_first_booking', 1, inplace=True)
    df.drop('date_account_created', 1, inplace=True)

def handle_age(df, age_data):
    def get_common_age():
        return 0

    df.fillna(get_common_age(), inplace=True)
   
def handle_gender(df, age_data):
    gen_dummies = pd.get_dummies(df['gender'])
    df.drop('gender', 1, inplace=True)
    gen_dummies.drop('-unknown-', 1, inplace=True)
    return df.join(gen_dummies)

if __name__ == '__main__':
    print('Starting feature engineering and data cleaning...')
    test_df = pd.read_csv('../data/airbnb-new-user/train_users.csv')
    print('Handling dates...')
    handle_dates(test_df)
    print('Finished dates...')

    age_data = pd.read_csv('../data/airbnb-new-user/age_gender_bkts.csv')
    print('Handling ages...')
    handle_age(test_df, age_data)
    print('Finished ages...')

    print('Handling gender...')
    test_df = handle_gender(test_df, age_data)
    print('Finished gender...')
