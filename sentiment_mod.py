# -*- coding: utf-8 -*-
"""sentiment_mod.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ONLyC-6Ki0n0qKYR2kO6r8uty-XDd_4w
"""

import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')

class VoteClassifier(ClassifierI):
  def __init__(self,*classifiers):
    self._classifiers=classifiers
  def classify(self,features):
    votes=[]
    for c in self._classifiers:
      v=c.classify(features)
      votes.append(v)
    return mode(votes)  
  def confidence(self,features):
    votes=[]
    for c in self._classifiers:
      v=c.classify(features)
      votes.append(v)
    choice_votes=votes.count(mode(votes))
    conf=choice_votes/len(votes)
    return conf

import zipfile
zipref=zipfile.ZipFile("../content/pickledalgos.zip")
zipref.extractall()
zipref.close()

documents_f=open("pickledalgos/documentsf.pickle","rb")
documents=pickle.load(documents_f)
documents_f.close()

wordfeatures_f=open("pickledalgos/wordfeatures_f.pickle","rb")
word_features=pickle.load(wordfeatures_f)
wordfeatures_f.close()

def find_features(document):
  words=word_tokenize(document)
  features={}
  for w in word_features:
    features[w]=( w in words)
  return features

featureset=open("pickledalgos/featureset.pickle","rb")
featuresets=pickle.load(featureset)
featureset.close()

random.shuffle(featuresets)
print(len(featuresets))

training_set=featuresets[:10000]
testing_set=featuresets[10000:]

open_file=open("pickledalgos/naivebayes.pickle","rb")
classifier=pickle.load(open_file)
open_file.close()

open_file=open("pickledalgos/MNB.pickle","rb")
MNB_classifier=pickle.load(open_file)
open_file.close()

open_file=open("pickledalgos/BNB.pickle","rb")
BNB_classifier=pickle.load(open_file)
open_file.close()

open_file=open("pickledalgos/LR.pickle","rb")
LogisticRegressionClassifier=pickle.load(open_file)
open_file.close()

open_file=open("pickledalgos/SGD.pickle","rb")
SGD_classifier=pickle.load(open_file)
open_file.close()

open_file=open("pickledalgos/NU.pickle","rb")
NU_classifier=pickle.load(open_file)
open_file.close()

open_file=open("pickledalgos/LSVC.pickle","rb")
LinearSVC_classifier=pickle.load(open_file)
open_file.close()

voted_classifier=VoteClassifier(classifier,
                                MNB_classifier,
                                BNB_classifier,
                                SGD_classifier,
                                LinearSVC_classifier,
                                LogisticRegressionClassifier,
                                NU_classifier)

def sentiment(text):
  feats=find_features(text)
  return voted_classifier.classify(feats),voted_classifier.confidence(feats)