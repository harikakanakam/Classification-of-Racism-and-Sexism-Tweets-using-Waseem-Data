import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

stemmer = SnowballStemmer("english")

with open('waseem/waseemtrain.txt', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')
    
train_data = pd.DataFrame({'text': lines})

train_labels = pd.read_csv('waseem/waseemtrainGold.txt', header=None, names=['target'])

pipeline = Pipeline([
    ('vect', TfidfVectorizer(tokenizer=lambda text: [stemmer.stem(word) for word in text.split()])),
    ('clf', LogisticRegression())
])

parameters = {
    'vect__max_df': [0.25, 0.5, 0.75],
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__C': [0.1, 1, 10],
}

grid_search = GridSearchCV(pipeline, parameters, cv=5)
grid_search.fit(train_data['text'], train_labels['target'])

print("Best parameters for logistic regression: ", grid_search.best_params_)
print("Best score for logistic regression: ", grid_search.best_score_)

with open('waseem/waseemtest.txt', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')
    
test_data = pd.DataFrame({'text': lines})
test_labels = pd.read_csv('waseem/waseemtestGold.txt', header=None, names=['target'])


predictions = grid_search.predict(test_data['text'])
print("Classification report for waseemtestGold: \n", classification_report(test_labels['target'], predictions))

test_data = pd.read_csv('hate_tweets.csv')

predictions = grid_search.predict(test_data['Tweet'])
print("Classification report for tweets from last 2-3 months: \n", classification_report(test_data['Label'], predictions))

