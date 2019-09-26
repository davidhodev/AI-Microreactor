from nltk import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,wordnet,WordNetLemmatizer
from collections import OrderedDict
import math
import operator
import os
import string
import numpy as np
from string import punctuation

''' Constants '''
path = "/home/benjamin/Python_Codes/AI-Microreactor/Abstracts/"
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
wordDictionary = {}

'''
Neural Network Class
'''
class NeuralNetwork():
    def __init__(self, lenofWordsDict):
        # seeding for random number generation
        np.random.seed(1)

        #converting weights to a lenofWordsDict by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((lenofWordsDict, 1)) - 1

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output

            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output
        #converting values to floats

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


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
    abstractFiles.sort()
    for file in abstractFiles:
        totalNumberOfAbstracts += 1
        infile = open(path+str(file))
        if totalNumberOfAbstracts < 5:
            print(str(file))
        abstract = infile.read()
        abstract = strip_punctuation(abstract)
        abstract = to_lower(abstract)
        listOfAllWords = word_tokenize(abstract)
        listOfAllWords = [lemmatizer.lemmatize(word) for word in listOfAllWords if not word in stop_words and len(word) > 2]
        for word in listOfAllWords:

            if word not in wordDictionary:
                wordDictionary[word] = [0]*(totalNumberOfAbstracts-1)
                wordDictionary[word].append(1)
            else:
                if len(wordDictionary[word]) < (totalNumberOfAbstracts):
                    wordDictionary[word] += 0 * ((totalNumberOfAbstracts) - len(wordDictionary[word]))
                #print(word, wordDictionary[word], totalNumberOfAbstracts, wordDictionary[word][totalNumberOfAbstracts-1])
                wordDictionary[word][totalNumberOfAbstracts-1] += 1

        for word in wordDictionary:
            if len(wordDictionary[word]) != totalNumberOfAbstracts+1:
                wordDictionary[word]+= [0]*(totalNumberOfAbstracts-len(wordDictionary[word])+1)

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
            tf = 0.5 + 0.5*(frequencyList[abstractFrequency] / maxOccurenceOfWord)
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
            totalDocuments = len(wordDictionary[word])-1 #[len(wordDictionary[word])-1][2]

            numberOfDocumentsWhereWordOccurs = wordDictionary[word][len(wordDictionary[word])-1][1]
            idf = math.log(totalDocuments/numberOfDocumentsWhereWordOccurs)
            tf = tf * idf
    return tfDictionary

#def getTfIdfPerAbstract

def tfidfDictTo2dArray(tfDictionary):
    output = []
    for key in tfDictionary:
        wordArray = [key]
        wordArray.append(tfDictionary[key])
        output.append(wordArray)
    return output

def main():
    wordDictionary = BuildDictionaryFromAbstracts()
    getMaxNumberAndNNZ(wordDictionary)
    tfDictionary = calculateTermFrequency(wordDictionary)
    tfDictionary = calculateTfIdf(tfDictionary, wordDictionary)
    inputArray = tfidfDictTo2dArray(tfDictionary)
    #print(tfDictionary["flow"])
    #print(wordDictionary["flow"])
    #plotWords(wordList, 75)
    #initializing the neuron class
    neural_network = NeuralNetwork(len(wordDictionary["flow"]))

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    #training data consisting of 4 examples--3 input values and 1 output
    inputTrainingArray = []
    for i in range(50): #First 20
        for k in range(1,len(inputArray[i])):
            inputTrainingArray.append(inputArray[i][k])

    print('LENGTH', len(inputTrainingArray[0]))
    print("INPUT TRAINING:", inputTrainingArray)
    training_inputs = np.array(inputTrainingArray)
    training_outputs = np.array([[1,1,0,1,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,1,1,1,1,0,1,1,1,1,0,1,0,0,1,0,1,0,1,1,0,1,1,1]]).T

    #training taking place
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    thinkArray = []
    for i in range(1,20):
        for k in range(1,len(inputArray[i])):
            thinkArray.append(inputArray[i][k])
        print(i, neural_network.think(np.array([thinkArray])))


if __name__ == "__main__":
    main()
