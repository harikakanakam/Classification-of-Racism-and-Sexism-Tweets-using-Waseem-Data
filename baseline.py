import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report

with open('waseem/waseemtrain.txt', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')
    
train_data = pd.DataFrame({'text': lines})

train_labels = pd.read_csv('waseem/waseemtrainGold.txt', header=None, names=['target'])

pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', SVC())
])

pipeline.fit(train_data['text'], train_labels['target'])

with open('waseem/waseemtest.txt', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')
    
test_data = pd.DataFrame({'text': lines})
test_labels = pd.read_csv('waseem/waseemtestGold.txt', header=None, names=['target'])

predictions = pipeline.predict(test_data['text'])
print("Classification report for waseemtestGold: \n", classification_report(test_labels['target'], predictions))

test_data = pd.read_csv('hate_tweets.csv')

predictions = pipeline.predict(test_data['Tweet'])
print("Classification report for tweets from last 2-3 months: \n", classification_report(test_data['Label'], predictions))
