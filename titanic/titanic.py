import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
     
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass     
     
    return False

def output_results(predicted):
    i = 892
    print("PassengerId,Survived")
    for p in predicted:
        print(str(i) +"," +str(p))
        i += 1

def clean_data(df):
    # drop columns we don't want to train on
    df.drop('PassengerId', 1, inplace=True)
    df.drop('Name', 1, inplace=True)
    df.drop('Ticket', 1, inplace=True)
    df.drop('Cabin', 1, inplace=True)

    # convert numbers to floats/ints
    df['Pclass'] = df['Pclass'].astype(int)
    
    average_age = df['Age'].mean()
    df['Age'][np.isnan(df['Age'])] = average_age
    df['Age'] = df['Age'].astype(int)

    df['SibSp'] = df['SibSp'].astype(int)
    df['Parch'] = df['Parch'].astype(int)

    df['has_family'] = [1 if row['SibSp'] > 0 or row['Parch'] > 0 else 0 for index, row in df.iterrows()]
    df.drop('SibSp', 1, inplace=True)
    df.drop('Parch', 1, inplace=True)
    
    avg_fare = df['Fare'].mean()
    df['Fare'][np.isnan(df['Fare'])] = avg_fare
    df['Fare'] = df['Fare'].astype(float)

    sex_dummies = pd.get_dummies(df['Sex'])
    df.drop('Sex', 1, inplace=True)
    df = df.join(sex_dummies)
    df.drop('male', 1, inplace=True)

    em_dummies = pd.get_dummies(df['Embarked'])
    #print em_dummies
    df.drop('Embarked', 1, inplace=True)
    df = df.join(em_dummies)

    return df

if __name__ == "__main__":
    test_data = pd.read_csv('../data/titanic/test.csv')
    train_data = pd.read_csv('../data/titanic/train.csv')

    train_df = clean_data(train_data)
    test_df = clean_data(test_data)
    
    # get our target data
    target = train_df['Survived'].astype(int)
    train_df.drop('Survived', 1, inplace=True)

    model =  GaussianNB()
    clf = model.fit(train_df, target)
    # score = clf.score(test_data, expected)
    # print (score)

    #with pd.option_context('display.max_rows', None, 'display.max_columns', 11):
     #   print(train_df)

    predicted = clf.predict(test_df)
    output_results(predicted)

