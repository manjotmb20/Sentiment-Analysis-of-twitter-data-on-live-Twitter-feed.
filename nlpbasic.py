# -*- coding: utf-8 -*-
"""NLPbasic.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g9SLaGrae5JhsJFge8vSsbE8emBKnD-i
"""

pip install nltk

import nltk
from nltk.corpus import stopwords

nltk.download()

from nltk.tokenize import sent_tokenize,word_tokenize

example_text="Hello, how are you. The weather is blue? You should not drink and drive. Awesome is python."

c=(sent_tokenize(example_text))

c

d=word_tokenize(example_text)

d

example_sent="This is an example of showing stopwords filteration"

stop_words=set(stopwords.words("english"))

words=word_tokenize(example_sent)

words

filtered=[]
for w in words:
  if w not in stop_words:
    filtered.append(w)

filtered

from nltk.stem import PorterStemmer

ps=PorterStemmer()

example_words=["python","pythons","pythoning"]

for w in example_words:
  print(ps.stem(w))

from nltk.corpus import state_union

from nltk.tokenize import PunktSentenceTokenizer

train_text=state_union.raw("2005-GWBush.txt")
sample_text=state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer=PunktSentenceTokenizer(train_text)

tokenized=custom_sent_tokenizer.tokenize(sample_text)

def process_content():
  try:
    for i in tokenized:
      words=nltk.word_tokenize(i)
      tagged=nltk.pos_tag(words)
      chunkGram= r"""Chunk: {<.?>+}   }<VB.?|IN|DT>+{"""
      chunkParser=nltk.RegexpParser(chunkGram)
      chunked=chunkParser.parse(tagged)
      print(chunked)
  except Exception as e:
     print(str(e))

import matplotlib.pyplot as plt

"""Part of speech tagging"""

def process_content2():
  try:
    for i in tokenized[5:]:
      words=nltk.word_tokenize(i)
      tagged=nltk.pos_tag(words)
      named_Ent= nltk.ne_chunk(tagged)
      named_Ent.draw()
  except Exception as e:
     print(str(e))

process_content2()

"""Lemmatization"""

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
print(lemmatizer.lemmatize("geese"))

"""wordnet"""

from nltk.corpus import wordnet

syns=wordnet.synsets("program")

syns[0]

print(syns[3].lemmas())

print(syns[0].lemmas()[1].name())

synonyms=[]
antonyms=[]

for syn in wordnet.synsets("good"):
  for l in syn.lemmas():
    synonyms.append(l.name())
    if l.antonyms():
      antonyms.append(l.antonyms()[0].name())

w1=wordnet.synset("ship.n.01")
w2=wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))

