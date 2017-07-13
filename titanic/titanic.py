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

if __name__ == "__main__":
    test_data = pd.read_csv('../data/titanic/test.csv')
    train_data = pd.read_csv('../data/titanic/train.csv')

    # drop columns we don't want to train on
    test_data.drop('PassengerId', 1, inplace=True)
    test_data.drop('Name', 1, inplace=True)
    test_data.drop('Ticket', 1, inplace=True)
    test_data.drop('Cabin', 1, inplace=True)

    train_data.drop('PassengerId', 1, inplace=True)
    train_data.drop('Name', 1, inplace=True)
    train_data.drop('Ticket', 1, inplace=True)
    train_data.drop('Cabin', 1, inplace=True)

    # convert numbers to floats/ints
    test_data['Pclass'] = test_data['Pclass'].astype(int)
    
    average_age = test_data['Age'].mean()
    test_data['Age'][np.isnan(test_data['Age'])] = average_age
    test_data['Age'] = test_data['Age'].astype(int)
    test_data['SibSp'] = test_data['SibSp'].astype(int)
    test_data['Parch'] = test_data['Parch'].astype(int)
    
    avg_fare = test_data['Fare'].mean()
    test_data['Fare'][np.isnan(test_data['Fare'])] = avg_fare
    test_data['Fare'] = test_data['Fare'].astype(float)

    sex_dummies = pd.get_dummies(test_data['Sex'])
    test_data.drop('Sex', 1, inplace=True)
    test_data.join(sex_dummies)

    em_dummies = pd.get_dummies(test_data['Embarked'])
    test_data.drop('Embarked', 1, inplace=True)
    test_data.join(em_dummies)

    train_data['Pclass'] = train_data['Pclass'].astype(int)
    
    average_age = train_data['Age'].mean()
    train_data['Age'][np.isnan(train_data['Age'])] = average_age
    train_data['Age'] = train_data['Age'].astype(int)
    train_data['SibSp'] = train_data['SibSp'].astype(int)
    train_data['Parch'] = train_data['Parch'].astype(int)
    train_data['Fare'] = train_data['Fare'].astype(float)

    sex_dummies = pd.get_dummies(train_data['Sex'])
    train_data.drop('Sex', 1, inplace=True)
    train_data.join(sex_dummies)

    em_dummies = pd.get_dummies(train_data['Embarked'])
    train_data.drop('Embarked', 1, inplace=True)
    train_data.join(em_dummies)
    
    # get our target data
    target = train_data['Survived'].astype(int)
    train_data.drop('Survived', 1, inplace=True)

    model =  RandomForestClassifier(n_estimators=100)
    clf = model.fit(train_data, target)
    # score = clf.score(test_data, expected)
    # print (score)

    predicted = clf.predict(test_data)
    output_results(predicted)

