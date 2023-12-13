# Kaggle-Project-NLP
![](UTA-DataScience-Logo.png)

# NLP disaster tweets kaggle competetion

* **One Sentence Summary** Ex: This repository holds an attempt to apply NLP to disaster tweets 
## Overview


  * **Definition of the tasks / challenge**  The goal of this competion is to use machine learning to detect whether a tweet is a natural disaster or not a disaster 
  * **Your approach** : This problem was formulated by predicting whether a tweet is a disaster (1) or not (2)- this is a binary classification problem. In order to acheieve this goal I used data reprocessing -tokenezation, stemming, count vectorazation to convert in numerical format- and bag of words which allowed me to convert text sequences.
  * **Summary of the performance achieved** : The linear regression model displayed a F1 score of 81% and the decesion tree model showed a f1 score of 77% .

## Summary of Workdone

Include only the sections that are relevant an appropriate.

### Data

* Data:
  
    * Input: 3 CSV files: sample_submission.csv, test.csv, train.csv
    * test.csv and train.csv has the following columns:
    * id= identifier for each tweet,
    * text= text of tweet
    * location= where tweet was sent from (may be blank)
    * keyword= particular keyword from tweet (may be blank
    * target- in train csv only denotes whether it is a real disaster(1) or not (0)  

* Size: 1.43 MB 
  * Instances: train data has 7614 ids (tweets) and test data has 3264 ids (tweets) 
    

#### Preprocessing / Clean up

* Tweets have many unneccesary words that is not needed for the purposes of this task. As such I used nltk to remove stopwords from the tweets. I also used countvectorazation, BOW, and TF-IDF to convert the text data into numerical format which is applicable to machine learning models.
* Once I compeletd these tasks I made a two new csv files test_clean_data.csv and train_clean_data.csv

#### Data Visualization

Here I can include unigrams, trigrams models I can also include most common words. Maybe even include cleaned test header from clean train data.? 

### Problem Formulation

* Define:
  * Input / Output: Input for this moedel is the tweet Output is 0/1
  * I used two models to analyze my results: Linear regression model to approach the model in a linear fashion. That is model assumed that there was a linear relation betwetn input features and targert variable. This is continous 
  * The secound model I chose was the decision tree model. This makes up for the problems from the linear regression model by not assuming a linear relationship. This model is both categorical and continous. 
     
  

### Training

* Describe the training:
  * The first step I did was clean the training data. Afterwards I used scikt-learn to convert the tweet into numerical format which is applicable to machine learning. The I split the into train and testing set which helped assed the models preformane on unseen data. Once I completed this I used linear regression and decesion tree to evaluate the model. 

### Performance Comparison

* First model is linear regression model. For the 0 tweets this model has 0.82 precesion, 0.87 recall, and 0.84 f1-score. For the 1 tweets this model has a 0.81 precesion, 0.74 recallm and a 0.77 f1 score. Overall this model is 81% accurate.
* The secound model is a decesion tree model. For the 0 tweets it has a 0.80 precision, 0.79 recall, and 0.80 f1 score. For the 1 tweets it had a 0.73 precision, 0.75 recall, and a 0.74 f1 score. Overall the model was 77% accurate. 

### Conclusions

* In conclusion my project uses data reprocessing, model selection, and evaulation metric to complete the goal of the challange: determining whether a tweet is disaster or non disaster

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * In order to reproduce the results please follow these instructions and follow the steps of "Kaggle Project notebook." Here is a general overview on how to reproduce results:
   * Download csv files ( train.csv, and test.csv)
   * Plot graphs that display what the data actually looks like. For example: plotting how many tweets are 1/0, common words in tweets, etc.
   * Clean data: remove stop words and apply stemming, HTML tags, remove nonalphabetic characters, lowercase, tokenize and join into string. Now you created new csv file 
   * Use clean csv file and implement count vectorizer and BOW- converting text to numerical format. Now you made bigrams, unigrams, trigrams. Plot for analysis
   * Now we can use TF-IDF to enhance bow
   * Now you can use train_test_split to split the data and you can print sentences and vectors
   * Now make linear regression model and decesion tree model using sckit learn 

     
    
   

### Overview of files in repository

* train_cleaned_data.csv: this is the clean training data
* test_cleaned_data.csv: this is the clean testing data
* Kaggle notebook: My analysis along with graphs I created 



### Software Setup
* In order to complete these tasks you need pandas, sckit learning, and nltk. You can download these by installing them in your bash/terminal. 



#### Performance Evaluation

* You can run performance evaluation by using sckit learn and splitting your data into training and testing. For this challange her is how I split data:
* training data used to train machine learning model
* validation makes sure machine learning model can be generalized
* testing is the final phase and it makes sure that the machine learning model is applicable 


## Citations
* Natural language processing with disaster tweets. Kaggle. (n.d.). https://www.kaggle.com/competitions/nlp-getting-started/data
  


* Provide any references.