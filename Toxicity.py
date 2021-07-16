# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:27:54 2020

@author: Rituparna
"""


import pandas as pd
from nltk.corpus import stopwords
import string
import re

#################### Q1. Importing Dataset ##################################

data = pd.read_csv("OneDrive\\Desktop\\Simplilearn\\NLP\\Projects\\Wikipedia toxicity\\train.csv")

data.info()

data.head(10)
data['comment_text'].head(10)

data.shape
# Out[3]: (5000, 3)

data.columns
# Out[4]: Index(['id', 'comment_text', 'toxic'], dtype='object')

data.toxic.value_counts()
# Out[5]: 
# 0    4563
# 1     437
# Name: toxic, dtype: int64
# From the above result it is clear that the data set is not balanced. The dataset is biased 
# towards the class 0, ie, non-toxic. the ratio of class distribution 
# is approx. 10:1 for non-toxic and toxic classes.

############# Q2. Converting the 'comment_text' into list ####################

comment_list = list(data['comment_text'])
# comment_list contains the data from the column 'comment_text' 
# which will be the feature for the classifier, ie, x.

toxicity = data.iloc[:, 2].values
print(toxicity)
# toxicity will serve as the label for the suprevised algorithm for the classification, ie, y.

#################### Q3-Q4. Removing IP address and URL ########################

rule_IP = r'\d{1,3}\.\d{1,4}\.\d{1,4}\.\d{1,4}' # RegEx for identifying IP addresses
rule_URL = r'http://\S+|https://\S+'            # RegEx for identifying URL

for comment in comment_list:
    IP = re.findall(rule_IP, comment)
    URL = re.findall(rule_URL, comment)
    if IP:
        print("IP: ", IP)
    if URL:
        print("URL ", URL)
        
# IP:  ['71.127.137.171']
# URL  ['http://en.wikipedia.org/wiki/Mutilation']
# IP:  ['72.75.20.29']
# IP:  ['89.241.146.140']
# URL  ['http://spanky.thehawkeye.com/features/IAAP/breaking/b1_0614.html', 'http://news.google.com/newspapers?nid=1350&dat;=19680608&id;=RGcxAAAAIBAJ&sjid;=mwEEAAAAIBAJ&pg;=2025,668035', 'http://www.brumm.com/genealogy/showmedia.php?mediaID=5646&medialinkID;=7442&PHPSESSID;=1bf779fbde1fec4673e3ec6b631deb7f']
# URL  ['http://www.valerosos.com/PreludetoInchon.html', 'http://www.valerosos.com/CommandsGVillahermosa.html', 'http://www.borinqueneers.com/node/245']
# IP:  ['42.60.139.23']
#..........................        
# Above code finds and prints all the IP addresses and URL from the comment list.        
        
    
refine_comment_list = []    # This empty list will contain the comments without IP or URL.

def remove_IP_URL():  
    for comment in comment_list:
        IP = re.findall(rule_IP, comment)
        URL = re.findall(rule_URL, comment)
        if URL:
            for each_URL in URL:
                comment = comment.replace(each_URL, '')
        if IP:
            for each_IP in IP:
                comment = comment.replace(each_IP, '')
        refine_comment_list.append(comment)
    
remove_IP_URL()

# Above function removes IP and URL from each and every comments from comment_list
# and store them in refine_comment_list.

for comment in refine_comment_list:
    IP = re.findall(rule_IP, comment)
    URL = re.findall(rule_URL, comment)
    if IP:
        print("IP: ", IP)
    if URL:
        print("URL ", URL)
        
# This function is to verify that the new list doesn't contain any IP or URL.
    
####################### Normalization and Tokenization ######################

from nltk.tokenize import word_tokenize

contextual_texts = ['wikipedia', 'edit', 'page', 'articles']

def text_processing(comment_text):
    remove_punctuation = [chars for chars in comment_text if chars not in string.punctuation]
    without_punctuation = ''.join(remove_punctuation)
    without_punctuation

    token_word = word_tokenize(without_punctuation)
    
    normalized_word = [words.lower() for words in token_word]
        
    processed_words = [words for words in normalized_word if words not in stopwords.words('english')]
                   
    final_words = [word for word in processed_words if word not in contextual_texts]   
    return(final_words)

# Above function does the text prepreocessing and also removes all the contextual text
# fron the final list of words.
    
#################### Q6. Creating TF-IDF Vector space ###############

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vector = TfidfVectorizer(analyzer = text_processing, max_features = 4000)
# Initializing the tfidfVectorizer with parameter, analyser as callable with 'text_processing'
# and vocabulory size of 4000 features.

print(tfidf_vector.vocabulary_)

final_vocab = tfidf_vector.fit_transform(refine_comment_list)
# Calculating tf-idf values for each features in the vocabulary which will be used as
# the features in the classifier model.

print(final_vocab)

count_token = tfidf_vector.get_feature_names()

len(count_token)
# Out[24]: 4000

################## Q5. Splitting into train and test data sets ######################################


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(final_vocab, toxicity,
                                                   test_size = 0.3, random_state = 60 )

# Using final_vocab as the dataset for the model building.
# Dataset is split into 70:30 ratio for training and test set respectively.

###################### Q7. Building SVM ######################################

from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

svm_cls = svm.SVC(kernel = 'linear')       # Initializing SVC with with kernel as 'Linear'.
svm_model = svm_cls.fit(x_train, y_train)  # Training the model 

svm_predict = svm_model.predict(x_test)

###################### Q8. Accuracy, recall, and f1_score #####################

svm_model.score(x_train, y_train) 
# Accuracy of training set :  Out[31]: 0.9654285714285714

svm_model.score(x_test, y_test)
# Accuracy of test set :  Out[32]: 0.9486666666666667

matrix_test = confusion_matrix(y_test, svm_predict)
print(matrix_test)
# [[1363    4]
#  [  73   60]]

report_test = classification_report(y_test, svm_predict)
print(report_test)
#              precision    recall  f1-score   support
# 
#            0       0.95      1.00      0.97      1367
#            1       0.94      0.45      0.61       133
# 
#     accuracy                           0.95      1500
#    macro avg       0.94      0.72      0.79      1500
# weighted avg       0.95      0.95      0.94      1500


matrix = confusion_matrix(y_train, svm_model.predict(x_train))
print(matrix)
# [[3208    2]
#  [ 109  181]]

report = classification_report(y_train, svm_model.predict(x_train))
print(report)
#               precision    recall  f1-score   support
# 
#            0       0.97      1.00      0.98      3210
#            1       0.99      0.62      0.77       290
#
#     accuracy                           0.97      3500
#    macro avg       0.98      0.81      0.87      3500
# weighted avg       0.97      0.97      0.96      3500

# As it can be seen from the above classification_reports of test set and training set,
# accuracy of training dataset is higher than the test data set. The model is performing better
# for training set. Model is dealing with over-fitting problem.
# Over-fitting problem can be due to the imbalance of classes of data in the data set
# which favoures class 0.

# In both cases the recall score is higher for class 0 which is expectable as a
# nontoxic comment shouldn't be label as toxic.

# lets try to overcome this problem by changing the parameter 'randon_state'
# and try to find parameter where the accuracy of test model is higher than the 
# train model.

################## Q9-Q10. Parameter Adjustment ###############################

def best_param():
    o = 0
    for i in range(1,101):
        x_train, x_test, y_train, y_test = train_test_split(final_vocab, toxicity,
                                                       test_size = 0.3, random_state=i)
        svm_cls.fit(x_train,y_train)
        
        trainScore = svm_cls.score(x_train,y_train)
        testScore = svm_cls.score(x_test,y_test)
        
        if testScore > trainScore and testScore > 0.9:
            o = 1
            print("Testing {} , Training {}, RS {}".format(testScore,trainScore,i))
        
    if o == 0:
        print('Best Parameter not found')

best_param()        
# Best Parameter not found
# No parameter is found where the the accuracy of test model is higher than the 
# train model. So, hypertunning of the parameter is required.

################# Q11-Q14. Hypertunning Parameter ###################

################# Using GridSearch #########################

from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {'C': [10, 100, 0.1, 1, 1000]}

cls_svm = svm.SVC(class_weight = 'balanced', kernel = 'linear')       
grid_search = GridSearchCV(cls_svm, param_grid, 
                           cv = StratifiedKFold(5), verbose = 3, scoring = 'recall')

grid_search_model = grid_search.fit(x_train, y_train)

grid_search_model.best_params_    
# Out[88]: {'C': 1000}

grid_search_model.best_estimator_
# Out[95]: 
# SVC(C=1000, break_ties=False, cache_size=200, class_weight='balanced',
#     coef0=0.0, decision_function_shape='ovr', degree=3, gamma='scale',
#     kernel='linear', max_iter=-1, probability=False, random_state=None,
#     shrinking=True, tol=0.001, verbose=False)


grid_search_pred = grid_search_model.predict(x_test)

grid_search_report = classification_report(y_test, grid_search_pred)
print(grid_search_report)
#                 precision    recall  f1-score   support
#
#            0       0.97      0.87      0.92      1367
#            1       0.36      0.74      0.48       133
#
#     accuracy                           0.86      1500
#    macro avg       0.66      0.80      0.70      1500
# weighted avg       0.92      0.86      0.88      1500

# Recall score for toxic comment is 0.74 or 74% which is acceptable.

############## Q15. Prominent terms in the toxic comments ##################

from sklearn.feature_extraction.text import CountVectorizer

l = tfidf_vector.inverse_transform(x_test) 

# Converting tf-idf values of x_test set into corresponding words.

lst = pd.DataFrame(columns = ['Words'])
lst['Words'] = l    
lst['toxic'] = grid_search_pred
lst

# Creating a dataframe with converted x_test data and predicted y data.

toxic_comment = lst[lst['toxic']==1]
toxic_comment_list = toxic_comment['Words']
toxic_comment_list 

# Separating comments for which class label is 1.
# 4             [absolute, account, fool, retard, retarded]
# 5       [baby, cause, cry, eyes, go, goodbye, hell, he...
# 20                                   [10, 2004, sep, utc]
# 29      [accusations, already, ask, basic, courtesy, d...
# 31                                                     []
#                       
# 1477    [200, around, column, curious, figure, forever...
# 1478    [edits, either, hypothesis, impact, keep, like...
# 1489                                      [cricket, team]
# 1498    [american, article, author, bother, difference...
# 1499                      [20, events, game, go, records]
# Name: Words, Length: 274, dtype: object

li = []
for each_list in toxic_comment_list:
    if each_list.size:
       str1 = ' '.join(each_list)
       li.append(str1)
li

# Transforming into corpus
# ['absolute account fool retard retarded',
#  'baby cause cry eyes go goodbye hell hes hey kiss listen love make might na near never sad see wan way wouldnt',
#  '10 2004 sep utc',
#  'accusations already ask basic courtesy dont even false fuck kindly make others people talk tell use vandalism',
#  'association girl',
#  'fair',
#  'article gay grounds like mad removal request would',
#  '200 accurately ago better described form years',
#  'ass dealing effective ones placing think way win yet',
# ........................................................
# ]

cv = CountVectorizer()
bow = cv.fit_transform(li)
print(bow)

# Using CountVectorizer() to create bag of words

toxic_vocab = cv.vocabulary_
print(toxic_vocab)

sum_words = bow.sum(axis=0)
print(sum_words)

# Adding the occurance of each word corpus

words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
print(words_freq)

# Mapping words with no. of occurance.

words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

# Sorting the mapped words to find the top most toxic words in the corpus.

toxic_top_15 = words_freq[0:16]

print(toxic_top_15)
#[('fuck', 23), ('like', 20), ('people', 18), ('talk', 18), ('fucking', 17), ('go', 16), 
# ('think', 16), ('im', 15), ('shit', 13), ('stupid', 12), ('youre', 12), ('know', 11), 
# ('cant', 11), ('get', 11), ('thats', 10), ('dont', 9)]
 
############################################################################################






