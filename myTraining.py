import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__== "__main__":

    # Read The Data
    df = pd.read_csv('C:\mystuff\data.csv')
    train, test = data_split(df, 0.2)
    X_train = train[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    X_test = test[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()

    Y_train = train[['infectionProb']].to_numpy().reshape(2312,)
    Y_test = test[['infectionProb']].to_numpy().reshape(577,)



    DecisionTree = DecisionTreeClassifier(max_depth=100, min_samples_split=4,
        random_state=0)
    #DecisionTree = MultiOutputClassifier(DecisionTree, n_jobs=-1)
    #scores = cross_val_score(DecisionTree, X, y, cv=5)
    #print('DecisionTree scores.mean: ', scores)

    DecisionTree = DecisionTree.fit(X_train, Y_train)
    pred = DecisionTree.predict(X_test)
    print('DecisionTree score: ',DecisionTree.score(X_train, Y_train))

    # open a file, where yu want to store the data
    file = open('model.pkl','wb')

    #dump information to that file
    pickle.dump(clf, file)
    file.close()
   

   