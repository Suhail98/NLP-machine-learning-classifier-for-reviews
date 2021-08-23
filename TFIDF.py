import os 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import array
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
#reading data and shuffle
def read_data():
    arr = os.listdir("review_polarity\\txt_sentoken\\neg")
    data = []
    for i in arr:
        f = open("review_polarity\\txt_sentoken\\neg\\" + i, encoding='utf-8')
        data += [f.read(),0]
        f.close()
    arr = os.listdir("review_polarity\\txt_sentoken\\pos")
    for i in arr:
        f = open("review_polarity\\txt_sentoken\\pos\\" + i, encoding='utf-8')
        data += [f.read(),1]
        f.close()
    data = np.array(data).reshape(2000,2)
    np.random.shuffle(data)
    return data
#
def split_data(data,n):
    data_train,lable_train = data[:n,0], data[:n,1]
    data_test,lable_test = data[n:,0], data[n:,1]
    return data_train,lable_train,data_test,lable_test


data = read_data()
nTrain = int(2000*.8)
data_train,label_train,data_test,label_test = split_data(data,nTrain)

vectorizer = TfidfVectorizer(stop_words='english')
dfidf_train = vectorizer.fit_transform(data_train)

df_idf=[]
for i in dfidf_train:
    df_idf.append(array(i.todense()).flatten().tolist())

clf = LogisticRegression(random_state=0,C=10).fit(df_idf, label_train)


dfidf_test = vectorizer.transform(data_test)

result = clf.predict(dfidf_test.todense())

count = 0
for i in range(len(result)):
    if result[i] == label_test[i]:
        count += 1
print("test accuracy = ",count / (2000-nTrain))

input_review = input("Enter your review: ")
dfidf_test = vectorizer.transform([input_review])
result = clf.predict(dfidf_test)
if result[0] == '0':
    print("negative")
else:
    print("positive")

