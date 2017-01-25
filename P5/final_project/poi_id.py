#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','deferral_payments','exercised_stock_options','bonus','restricted_stock','restricted_stock_deferred','director_fees','long_term_incentive','deferred_income','expenses','bonus'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict['TOTAL']
data_dict['BELFER ROBERT']['deferral_payments'] = 102500
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
del data_dict['THE TRAVEL AGENCY IN THE PARK']
for key, value in data_dict.iteritems():
    for feature, values in value.iteritems():
        if feature in features_list:
            if values == 'NaN':
                data_dict[key][feature] = 0
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler

#Defining the Stratified Shuffle Split
sss = StratifiedShuffleSplit(
    n_splits = 1000,
    test_size=0.1,
    train_size=None,
    random_state=42)

def grid_search(features, labels):
    pipeline1 = Pipeline((
    ('scale',MinMaxScaler()),
    ('kbest', SelectKBest()),
    ('kneighbors', KNeighborsClassifier()),
    ))

    parameters1 = {
    'kneighbors__n_neighbors': [3, 7, 10],
    'kneighbors__weights': ['uniform', 'distance'],
    'kbest__k': [3,5,10]
    }

    print "starting Gridsearch"
    gs = GridSearchCV(pipeline1, parameters1, scoring = 'f1', cv= sss, n_jobs=-1)
    gs = gs.fit(features, labels)
    print "finished pipeline Gridsearch"
    print("The best parameters are %s with a score of %0.2f" % (gs.best_params_, gs.best_score_))
    clf = gs.best_estimator_
    return clf

clf = grid_search(features,labels)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
