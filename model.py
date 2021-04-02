"""
Author: Zitong Wu
Date: Nov. 1, 2020

Description:
Predict if a product on Amazon is awesome or not
Product type: grocery and gourmet food
Awesomeness threshold: awesome if predicted rating > 4.4 (out of 5)

Steps:
1. Preprocess the training data (clean the review and summary entries)
2. Generate review and summary features
3. Fit a logisitic regression model on the training data
4. Make predictions on the test data.  

Running the following code will print the accuracy, F1 and AUC score obtained from 
a 10-fold cross validation of the model, as well as outputting the predictions to a csv file

"""


import pandas as pd
import numpy as np
from nltk import download
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# download the required nltk resources
download('stopwords')
download('punkt')
pd.options.mode.chained_assignment = None 


## PREPARE TRAINING DATA!
# read in training data
df = pd.read_json('Grocery_and_Gourmet_Food_Reviews_training.json', lines=True)
print('Dimension of training data:', df.shape)
print('Number of unique products:', len(df.asin.unique()))
print()

# drop rows with missing asin, overall, reviewID, reviewText
df = df.dropna(subset = ['asin', 'overall', 'reviewerID', 'reviewText'])

# preprocess text in training data
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
def prep(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if not token in stop_words]
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)
df['cleanReviewText'] = df['reviewText'].apply(prep)
df['cleanSummary'] = df['summary'].fillna("")
df['cleanSummary'] = df['cleanSummary'].apply(prep)    



# K-fold Cross Validation
acurracy_all = 0
f1_all = 0
num_split = 10
kf = KFold(n_splits = num_split)
iteration = 1

for train_index, valid_index in kf.split(df):
    # split training data into train and validation set 
    X_train, X_valid = df.iloc[train_index], df.iloc[valid_index]
    y_train, y_valid = df.iloc[train_index], df.iloc[valid_index]  
    
    # generate review and summary features for the training set
    vectorizer_review = TfidfVectorizer()
    vectorizer_summary = TfidfVectorizer()
    X_train_review = vectorizer_review.fit_transform(X_train['cleanReviewText'])
    X_train_summary = vectorizer_summary.fit_transform(X_train['cleanSummary'])
    X_train_text = sp.hstack((X_train_review,X_train_summary))

    # generate review and summary features for the validation set
    X_valid_review = vectorizer_review.transform(X_valid['cleanReviewText']) 
    X_valid_summary = vectorizer_summary.transform(X_valid['cleanSummary'])
    X_valid_text = sp.hstack((X_valid_review,X_valid_summary)) 
              
    # train
    model = LogisticRegression(max_iter=1000, n_jobs=-1, solver='saga')
    model.fit(X_train_text, y_train['overall'])
    
    # validate
    y_valid_predict = model.predict_proba(X_valid_text) 
    y_valid_predict = y_valid_predict * np.array([1,2,3,4,5])
    y_valid_predict = np.sum(y_valid_predict, axis = 1)
    y_valid['predict'] = y_valid_predict

    y_valid_product = y_valid.groupby(['asin']).mean()
    y_valid_product['awesome'] = y_valid_product['overall'].apply(lambda x: 1 if x > 4.4 else 0)
    y_valid_product['predict'] = y_valid_product['predict'].apply(lambda x: 1 if x > 4.4 else 0)
    a = accuracy_score(y_valid_product['awesome'], y_valid_product['predict'])
    f = f1_score(y_valid_product['awesome'], y_valid_product['predict'])
    print("Iteration: " + str(iteration))
    print("Model: Logistic Regression")
    print("F1: " + str(f))
    print("Accuracy: " + str(a))
    print()
    
    acurracy_all += a
    f1_all += f  
    iteration += 1
    
print("10-Fold Cross Validation Results")
print("Model: Logistic Regression")
print("F1: " + str(f1_all/num_split))
print("Accuracy: " + str(acurracy_all/num_split))
print()



## PREDICT!
df_test = pd.read_json('Grocery_and_Gourmet_Food_Reviews_test.json', lines=True)
print('Dimension of test data:', df_test.shape)
print('Number of unique products:', len(df_test.asin.unique()))
print()

# preprocess text in test data
df_test = df_test.dropna(subset = ['asin', 'reviewerID', 'reviewText'])
df_test['cleanReviewText'] = df_test['reviewText'].apply(str).apply(prep)
df_test['cleanSummary'] = df_test['summary'].apply(str).apply(prep)
X_test_review = vectorizer_review.transform(df_test['cleanReviewText']) 
X_test_summary = vectorizer_summary.transform(df_test['cleanSummary'])
X_test = sp.hstack((X_test_review,X_test_summary)) 

# make predictions for test data
df_test['prediction'] = model.predict(X_test)
df_test_product = df_test.groupby(['asin']).mean()
df_test_product = df_test_product.reset_index()
df_test_product['prediction'] = df_test_product['prediction'].apply(lambda x: 1 if x > 4.4 else 0)
output = df_test_product[['asin', 'prediction']].sort_values('asin')
print("Predictions on Test Data:")
print("Awesome is 1, not awesome is 0")
print(output)

# Output predictions to csv file
output.to_csv("final_predictions_" + str(num_split) + "fold.csv")

