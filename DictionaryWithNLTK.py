from nltk import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,wordnet,WordNetLemmatizer
from collections import OrderedDict
import math
import operator
import os
import string
from string import punctuation

''' Constants '''
path = "/home/benjamin/Python_Codes/AI-Microreactor/Abstracts/"
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
wordDictionary = {}


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
def BuildDictionaryFromAbstracts():
    totalNumberOfAbstracts = 0
    wordList = []
    abstractFiles = os.listdir('Abstracts/')
    for file in abstractFiles:
        totalNumberOfAbstracts += 1
        infile = open(path+str(file))
        abstract = infile.read()
        abstract = strip_punctuation(abstract)
        abstract = to_lower(abstract)
        listOfAllWords = word_tokenize(abstract)
        listOfAllWords = [lemmatizer.lemmatize(word) for word in listOfAllWords if not word in stop_words and len(word) > 2]
        for word in listOfAllWords:
            if word not in wordDictionary:
                wordDictionary[word] = [0]*(totalNumberOfAbstracts-1) + [1]
            else:
                if len(wordDictionary[word]) < (totalNumberOfAbstracts):
                    wordDictionary[word] += [0] * ((totalNumberOfAbstracts) - len(wordDictionary[word]))
                wordDictionary[word][totalNumberOfAbstracts-1] += 1


        wordList = wordList + listOfAllWords
        infile.close()
    #print(wordDictionary["regime"])
    #count = 0
    #for i in wordDictionary["regime"]:
    #    count += i
    #print("COUNT: ", count)
    return wordDictionary

    #return wordList, totalNumberOfAbstracts


'''
Plots all words on a graph
'''
def plotWords(wordList, numberOfWordsPlotted = 50):
    largeString = FreqDist(wordList)
    largeString.plot(numberOfWordsPlotted)
    with open("dictionarynltk.txt", 'w') as f:
    	print(largeString, file=f)


'''
Adds the Max number of times a word appears in any one abstract
Also counts the number of abstracts the word appears in
Returns a dictionary with key as the word and value as the list
0: Most number of words used in an abstract
1: Number of documents word occurs in
2: Total frequency of word
'''
def getMaxNumberAndNNZ(wordDictionary):
    for word in wordDictionary:
        appendingList = [max(wordDictionary[word]), ((len(wordDictionary[word])) - wordDictionary[word].count(0)), sum(wordDictionary[word])]
        wordDictionary[word].append(appendingList)

'''
Calculates the tf(t,d) Term Frequency
Uses Augmented Frequency to prevent a bias towards longer documents
Outputs new Dictionary of all the tf's per word per abstract
'''
def calculateTermFrequency(wordDictionary):
    outputDictionary = {}
    for word in wordDictionary:
        frequencyList = wordDictionary[word]
        termFrequencyPerWord = []
        for abstractFrequency in range(len(frequencyList)-1):
            maxOccurenceOfWord = frequencyList[len(frequencyList)-1][0]
            tf = 0.5*(frequencyList[abstractFrequency] / maxOccurenceOfWord) + 0.5
            termFrequencyPerWord.append(tf)
        outputDictionary[word] = termFrequencyPerWord
    return outputDictionary

'''
Calculates the Inverse Document Frequency
Measures how much information the word provides
Inputs the term Frequency list
Outputs the dictionary completed with the Tf-Idfs
'''
def calculateTfIdf(tfDictionary,wordDictionary):
    for word in tfDictionary:
        for tf in range(len(tfDictionary[word])-1):
            totalDocuments = wordDictionary[word][len(wordDictionary[word])-1][2]
            print(wordDictionary[word][len(wordDictionary[word])-1])
            numberOfDocumentsWhereWordOccurs = wordDictionary[word][len(wordDictionary[word])-1][1]
            idf = math.log(totalDocuments/numberOfDocumentsWhereWordOccurs)
            tf = tf * idf
    return tfDictionary



def main():
    wordDictionary = BuildDictionaryFromAbstracts()
    getMaxNumberAndNNZ(wordDictionary)
    tfDictionary = calculateTermFrequency(wordDictionary)
    tfDictionary = calculateTfIdf(tfDictionary, wordDictionary)
    print(tfDictionary["flow"])
    print(wordDictionary["flow"])
    #print(wordDictionary["flow"])
    #plotWords(wordList, 75)

if __name__ == "__main__":
    main()
