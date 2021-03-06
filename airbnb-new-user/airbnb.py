import random
from dateutil.parser import parse

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

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
            days_to_book.append(0)
            season.append(0)
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

def handle_lang(df):
    lang_dummies = pd.get_dummies(df['language'])
    df.drop('language', 1, inplace=True)
    return df.join(lang_dummies)

def clean_unwanted_columns(df):
    df.drop('signup_method', 1, inplace=True)
    df.drop('signup_flow', 1, inplace=True)
    df.drop('affiliate_channel', 1, inplace=True)
    df.drop('affiliate_provider', 1, inplace=True)
    df.drop('first_affiliate_tracked', 1, inplace=True)
    df.drop('signup_app', 1, inplace=True)
    df.drop('first_device_type', 1, inplace=True)
    df.drop('first_browser', 1, inplace=True)

def test_locally(df):
    df.drop('id_', 1, inplace=True)
    test_df = df['country_destination']
    df.drop('country_destination', 1, inplace=True)

    X_train, X_test, Y_train, Y_test = train_test_split(
        df, test_df, test_size=0.4, random_state=0)

    clf = GaussianNB()
    clf = clf.fit(X_train, Y_train)
    print('Gaussian: ' +str(clf.score(X_test, Y_test)))
    
    clf = MultinomialNB()
    clf = clf.fit(X_train, Y_train)
    print('Multinomial: ' +str(clf.score(X_test, Y_test)))

    clf = RandomForestClassifier()
    clf = clf.fit(X_train, Y_train)
    print('Random Forest Classifier: ' +str(clf.score(X_test, Y_test)))

def process_data(df):
    print('Handling dates...')
    handle_dates(df)
    print('Finished dates...')

    age_data = pd.read_csv('../data/airbnb-new-user/age_gender_bkts.csv')
    print('Handling ages...')
    handle_age(df, age_data)
    print('Finished ages...')

    print('Handling gender...')
    df = handle_gender(df, age_data)
    print('Finished gender...')
        
    print('Handling language...')
    df = handle_lang(df)
    print('Finished language...')

    print('Removing unused columns')
    clean_unwanted_columns(df)
    print('Columns removed')

    return df

def output_predictions(train_df, test_df):
    train_df.drop('id_', 1, inplace=True)
    train_Y = df['country_destination']
    train_df.drop('country_destination', 1, inplace=True)

    clf = RandomForestClassifier()
    clf = clf.fit(train_df, train_Y)

if __name__ == '__main__':
    print('Starting feature engineering and data cleaning...')
    train_df = pd.read_csv('../data/airbnb-new-user/train_users.csv')
    train_df = process_data(train_df)
    # test_locally(train_df)
    
    print('Outputting predictions')
    # test_df = pd.read_csv('../data/airbnb-new-user/test_users.csv')
    # output_predictions(train_df, test_df)
    print('Finished')

