import csv

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

def get_training_data():
    training_data = []
    target = []
    with open('../data/titanic/train.csv', 'r') as csvfile:
        rows = csv.reader(csvfile)
        features = [2, 4, 5, 6, 7, 9, 11] #index of features we want to train with
        for i, row in enumerate(rows):
            if i == 0:
                continue
            training_row = []
            for feature in features:
                if not is_number(row[feature]):
                    training_row.append(abs(hash(row[feature])))
                else:
                    training_row.append(float(row[feature]))    
            training_data.append(training_row)
            target.append(row[1])

    return training_data, target

def get_testing_data():
    target = []
    testing_data = []
    with open('../data/titanic/test.csv', 'r') as csvfile:
        rows = csv.reader(csvfile)
        features = [1, 3, 4, 5, 6, 8, 10] #index of features we want to train with
        for i, row in enumerate(rows):
            if i == 0:
                continue
            testing_row = []
            for feature in features:
                if not is_number(row[feature]):
                    testing_row.append(abs(hash(row[feature])))
                else:
                    testing_row.append(float(row[feature]))
            testing_data.append(testing_row)

    with open('../data/titanic/gender_submission.csv', 'r') as csvfile:
        rows = csv.reader(csvfile)
        for i, row in enumerate(rows):
            if i == 0:
                continue
            target.append(row[1])

    return testing_data, target    

def output_results(predicted):
    i = 892
    print("PassengerId,Survived")
    for p in predicted:
        print(str(i) +"," +str(p))
        i += 1

if __name__ == "__main__":
    train_data, target = get_training_data()
    test_data, expected = get_testing_data()
    
    model =  RandomForestClassifier(n_estimators=100)
    clf = model.fit(train_data, target)
    score = clf.score(test_data, expected)
    print (score)

    # predicted = clf.predict(test_data)
    # output_results(predicted)

