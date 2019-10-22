from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import glob

import matplotlib.pyplot as plt

import json

train_sentiment_score = {}
for name in glob.glob('petfinder-adoption-prediction/train_sentiment/*'):
    with open(name) as data_file:
        ans = json.load(data_file)
    sentence_score = 0
    for item in ans['sentences']:
        sentence_score += item['sentiment']['score']

    train_sentiment_score[name.split('/')[-1][0:-5]] = {}
    train_sentiment_score[name.split('/')[-1][0:-5]]['magnitute'] = ans['documentSentiment']['magnitude']
    train_sentiment_score[name.split('/')[-1][0:-5]]['score'] = ans['documentSentiment']['score']
    train_sentiment_score[name.split('/')[-1][0:-5]]['sentence_score'] = sentence_score

train_sentiment_df =  DataFrame.from_dict(train_sentiment_score, orient='index')
train_sentiment_df.reset_index(inplace=True)
train_sentiment_df.columns = ['PetID','magnitude', 'score', 'sentence_score']
print(train_sentiment_df.head())

test_sentiment_score = {}
for name in glob.glob('petfinder-adoption-prediction/test_sentiment/*'):
    with open(name) as data_file:
        ans = json.load(data_file)
    test_sentiment_score[name.split('/')[-1][0:-5]] = {}
    test_sentiment_score[name.split('/')[-1][0:-5]]['magnitute'] = ans['documentSentiment']['magnitude']
    test_sentiment_score[name.split('/')[-1][0:-5]]['score'] = ans['documentSentiment']['score']
    test_sentiment_score[name.split('/')[-1][0:-5]]['scaled_score'] = ans['documentSentiment']['score']*ans['documentSentiment']['magnitude']

test_sentiment_df =  DataFrame.from_dict(test_sentiment_score, orient='index')
test_sentiment_df.reset_index(inplace=True)
test_sentiment_df.columns = ['PetID','magnitude','score', 'sentence_score']
print(test_sentiment_df.head())



train_raw = pd.read_csv("petfinder-adoption-prediction/train.csv", index_col = None)
train_raw =pd.merge(train_raw, train_sentiment_df, on = 'PetID')

train_raw = train_raw[train_raw['Type'] == 2]

pred_raw = pd.read_csv("petfinder-adoption-prediction/test/test.csv", index_col = None)
pred_raw =pd.merge(pred_raw, test_sentiment_df, on = 'PetID')
pred_raw = pred_raw[pred_raw['Type'] == 2]
breed_labels = pd.read_csv("petfinder-adoption-prediction/breed_labels.csv", index_col = None)
color_labels = pd.read_csv("petfinder-adoption-prediction/color_labels.csv", index_col = None)
with open ("petfinder-adoption-prediction/rating.json") as rating_json:
    rating = json.load(rating_json)
cat_rating_df = DataFrame.from_dict(rating['cat_breeds'], orient = 'index')
cat_rating_df.reset_index(inplace=True)
print(cat_rating_df.columns)
cat_rating_df.columns = ['BreedName', 'Affectionate with Family', 'Amount of Shedding',
       'Easy to Groom', 'General Health', 'Intelligence', 'Kid Friendly',
       'Pet Friendly', 'Potential for Playfulness',
       'Friendly Toward Strangers', 'Tendency to Vocalize']

dog_rating_df = DataFrame.from_dict(rating['dog_breeds'], orient = 'index')
dog_rating_df.reset_index(inplace=True)
print(dog_rating_df.columns)
dog_rating_df.columns = ['BreedName', ' Adaptability', ' All Around Friendliness', ' Exercise Needs',
       ' Health Grooming', ' Trainability', 'Adapts Well to Apartment Living',
       'Affectionate with Family', 'Amount Of Shedding', 'Dog Friendly',
       'Drooling Potential', 'Easy To Groom', 'Easy To Train', 'Energy Level',
       'Exercise Needs', 'Friendly Toward Strangers', 'General Health',
       'Good For Novice Owners', 'Incredibly Kid Friendly Dogs',
       'Intelligence', 'Intensity', 'Potential For Mouthiness',
       'Potential For Playfulness', 'Potential For Weight Gain', 'Prey Drive',
       'Sensitivity Level', 'Size', 'Tendency To Bark Or Howl',
       'Tolerates Being Alone', 'Tolerates Cold Weather',
       'Tolerates Hot Weather', 'Wanderlust Potential']

cat_breed_labels = breed_labels[breed_labels['Type'] == 2]
dog_breed_labels = breed_labels[breed_labels['Type'] == 1]

cat_breed_rating_df = pd.merge(cat_breed_labels, cat_rating_df, on = 'BreedName', how = 'left')
cat_breed_rating_df = cat_breed_rating_df.fillna(0)

dog_breed_rating_df = pd.merge(dog_breed_labels, dog_rating_df, on = 'BreedName', how = 'left')
dog_breed_rating_df = dog_breed_rating_df.fillna(0)
print(train_raw.describe())
print(cat_breed_rating_df.describe())
train_raw = pd.merge(train_raw, cat_breed_rating_df, left_on = ['Type', 'Breed1'], right_on = ['Type','BreedID'])
pred_raw = pd.merge(pred_raw, cat_breed_rating_df, left_on = ['Type', 'Breed1'], right_on = ['Type','BreedID'])

print(train_raw.columns)
print(breed_labels.columns)
print(color_labels.columns)
print(pred_raw.columns)
train_raw['breed_score'] = train_raw['Affectionate with Family'] + train_raw['Amount of Shedding'] + train_raw['Easy to Groom'] + train_raw['General Health'] + train_raw['Intelligence'] + train_raw[ 'Kid Friendly'] + train_raw['Pet Friendly'] + train_raw['Potential for Playfulness'] + train_raw['Friendly Toward Strangers'] + train_raw['Tendency to Vocalize']

pred_raw['breed_score'] = pred_raw['Affectionate with Family'] + pred_raw['Amount of Shedding']+ pred_raw['Easy to Groom']+ pred_raw['General Health']+ pred_raw['Intelligence']+ pred_raw[ 'Kid Friendly']+ pred_raw['Pet Friendly'] + pred_raw['Potential for Playfulness']+ pred_raw['Friendly Toward Strangers'] + pred_raw['Tendency to Vocalize']

dataTrain = train_raw[['Type', 'Name', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID',
       'VideoAmt', 'Description', 'PetID', 'PhotoAmt', 'AdoptionSpeed',
       'magnitude', 'score', 'sentence_score', 'BreedID', 'BreedName',
       'breed_score']]
dataPred = pred_raw[['Type', 'Name', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID',
       'VideoAmt', 'Description', 'PetID', 'PhotoAmt',
       'magnitude', 'score', 'sentence_score', 'BreedID', 'BreedName',
       'breed_score']]

dataTrain = dataTrain.sample(n = int(len(dataTrain)))
dataPred = dataPred.sample(n = int(len(dataPred)))
print(dataTrain.describe())
n = len(dataTrain)
train_start = 0
ratio = 1
train_end = int(np.floor(0.8*n)*ratio)
test_start = int(np.floor(0.8*n))
test_end = n
print("OK")
data_train = dataTrain[train_start:train_end]
data_test = dataTrain[test_start:test_end]
print("OK")

from pandas.plotting import scatter_matrix
feature_list = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
       'VideoAmt', 'PhotoAmt']
train = data_train[feature_list]
test = data_test[feature_list]
pred_data = dataPred[feature_list]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train)

#scaler.fit(data_test)
train = scaler.transform(train)
test = scaler.transform(test)
pred_data = scaler.transform(pred_data)

train = DataFrame(train, columns = feature_list)
test = DataFrame(test, columns = feature_list)
pred_data = DataFrame(pred_data, columns = feature_list)
#Build X and y
X_train = train
y_train = data_train["AdoptionSpeed"]

X_test = test
y_test = data_test["AdoptionSpeed"]

X_pred = pred_data

rnd_clf = RandomForestClassifier()
knn_clf = KNeighborsClassifier(n_neighbors = 5)
svm_clf = SVC(kernel = 'rbf', gamma = 0.1, C = 100)
"""
voting_clf = VotingClassifier(
     estimators = [('svc', svm_clf), ('rf', rnd_clf), ('knn', knn_clf)], voting = 'hard'
)
voting_clf.fit(X_train, y_train)

predict = voting_clf.predict(X_pred)

clf_name = ["SVM"]
clf_list = [svm_clf]

clf_name = ["KNN"]
clf_list = [knn_clf]
"""
clf_name = ["RF"]
clf_list = [rnd_clf]

for i in range(0, len(clf_name)):
    clf = clf_list[i]
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_predict))
    clf.score(X_test, y_test)

    feature_importances = pd.DataFrame(clf.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)

    predict = clf.predict(X_pred)
    con_matrix = confusion_matrix(y_test, y_predict)
    print(con_matrix)
    results = DataFrame({"PetID": dataPred.PetID, "AdoptionSpeed": predict})
    results.to_csv(clf_name[i] + "_results.csv", index=False)



