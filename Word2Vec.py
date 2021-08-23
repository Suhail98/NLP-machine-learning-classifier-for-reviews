import os 
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import array
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
from nltk.corpus import stopwords
import re
import random 

warnings.filterwarnings(action = 'ignore')
  
import gensim
from gensim.models import Word2Vec
random.seed(123)
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

def prepare(data):
    allData = []
    for review in data:
        revArray = review.split(" ")
        allData.append(revArray)
    return allData

def prepareDataClass(data,model,size):
    res=[]
    for i in data:
        sum=[0]*size
        sum=np.array(sum)
        num=0
        for j in i:
            if j in model.wv.key_to_index:
                tem = model.wv[j]
                tem = np.array(tem)
                num+=1
                sum = np.add(sum,tem)
        res.append(sum/num)
    return res

data = read_data()
nTrain = int(2000 * 0.8)

data_train,label_train,data_test,label_test = split_data(data,nTrain)
data_train = prepare(data_train)
data_test = prepare(data_test)
#print(data_train)
vector_size = 100
model = gensim.models.Word2Vec(data_train, min_count = 5, vector_size = vector_size, window = 10, sg = 1)
#clean data
data_train = prepareDataClass(data_train,model,vector_size)
data_test = prepareDataClass(data_test,model,vector_size)

gnb = GaussianNB()
result_NB = gnb.fit(data_train, label_train).predict(data_test)

clf = LogisticRegression(random_state=0,C=100).fit(data_train, label_train)
result_Logistic = clf.predict(data_test)

count_NB = 0
count_Log = 0
for i in range(len(result_NB)):
    if result_NB[i] == label_test[i]:
        count_NB += 1
    if result_Logistic[i] == label_test[i]:
        count_Log += 1
        
print("test accuracy for NB = ",count_NB / (2000-nTrain))
print("test accuracy for Logistic = ",count_Log / (2000-nTrain))    