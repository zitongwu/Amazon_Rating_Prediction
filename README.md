# Amazon_Rating_Prediction    
### Zitong Wu, November 2020
<br >  

## Description
Create a binary classifier to predict if a product on Amazon is awesome or not based on reviews.   
Awesomeness threshold: awesome if predicted rating > 4.4 (out of 5)

## Perfomance
The trained model obtained an F1 score of 0.88 on the validation set (with 10-fold cross validation) and 0.86 on the test set

## Data Size
* Training set: 921782 review entries, 33056 unique products  
* Test set: 222078 review entries, 8264 unique products

## General Approach 
* First make a multi-class (1,2,3,4,5) rating prediction for each review
* Then average the rating predictions for each product
* Finally classify the product as "awesome" if average rating > 4.4

:star2: Insight: Making a multilcass prediction as an intermediate step for binary classification improved the F1 performance by 6 percent 

## Steps
1. Preprocess the data (clean the review and summary entries)
2. Generate review and summary features
3. Fit a logisitic regression model on the training set
4. Validate the model performance with the validation set
5. Make predictions on the test set
