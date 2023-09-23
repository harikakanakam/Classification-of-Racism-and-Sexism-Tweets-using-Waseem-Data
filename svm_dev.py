import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

with open('waseem/waseemtrain.txt', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')
    
train_data = pd.DataFrame({'text': lines})
train_labels = pd.read_csv('waseem/waseemtrainGold.txt', header=None, names=['target'])

pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', SVC())
])

# Use GridSearchCV to find the best hyperparameters
parameters = {
    'vect__ngram_range': [(1,1), (1,2)],
    'vect__max_df': [0.5, 0.75, 1.0],
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf']
}
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
grid_search.fit(train_data['text'], train_labels['target'])

print("Best parameters for SVM:", grid_search.best_params_)
print("Best score for SVM:", grid_search.best_score_)


# Use the fitted pipeline to predict on the test data
test_data = pd.DataFrame({'text': lines})
test_labels = pd.read_csv('waseem/waseemtestGold.txt', header=None, names=['target'])

test_data_transformed = grid_search.best_estimator_.named_steps['vect'].transform(test_data)
test_score = grid_search.best_estimator_.named_steps['clf'].score(test_data_transformed, test_labels)
print("Test score for waseemtestGold: ", test_score)

test_data = pd.read_csv('hate_tweets.csv')
test_labels = test_data['Label']
test_data = test_data['Tweet']
test_data_transformed = grid_search.best_estimator_.named_steps['vect'].transform(test_data)
test_score = grid_search.best_estimator_.named_steps['clf'].score(test_data_transformed, test_labels)
print("Test score for tweets from last 2-3 months:", test_score)
