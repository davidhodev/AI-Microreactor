from nltk import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,wordnet,WordNetLemmatizer
from collections import OrderedDict
import operator
import os
import string
from string import punctuation

''' Constants '''
path = "/home/benjamin/Python_Codes/AI-Microreactor/Abstracts/"
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


'''
Turns all text into lower case
Returns single string
'''
def to_lower(text):
    return ' '.join([w.lower() for w in word_tokenize(text)])


'''
Strips all abstracts of punctuations
Return single string
'''
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)


'''
Gets all abstracts from a directory
Lemmonizes all Abstracts
Returns a list of words
'''
def getAbstractAndLemmonize():
    wordList = []
    abstractFiles = os.listdir('Abstracts/')
    for file in abstractFiles:
    	infile = open(path+str(file))
    	abstract = infile.read()
    	abstract = strip_punctuation(abstract)
    	abstract = to_lower(abstract) # makes lowercase
    	listOfAllWords = word_tokenize(abstract) # makers into l
    	listOfAllWords = [lemmatizer.lemmatize(word) for word in listOfAllWords]
    	filteredWords = [word for word in listOfAllWords if not word in stop_words and len(word) > 2] # filters out stopwords
    	wordList = wordList + filteredWords
    	infile.close()
    return wordList


'''
Plots all words on a graph
'''
def plotWords(wordList, numberOfWordsPlotted = 50):
    largeString = FreqDist(wordList)
    largeString.plot(numberOfWordsPlotted)
    with open("dictionarynltk.txt", 'w') as f:
    	print(largeString, file=f)

def main():
    wordList = getAbstractAndLemmonize()
    plotWords(wordList, 75)

if __name__ == "__main__":
    main()
