from nltk import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,wordnet,WordNetLemmatizer
from collections import OrderedDict
import operator
import os
import string
from string import punctuation
path = "/home/benjamin/Python_Codes/AI-Microreactor/Abstracts/"
savepath = "/home/benjamin/Python_Codes/AI-Microreactor/Dictionary/"
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def to_lower(text):
    return ' '.join([w.lower() for w in word_tokenize(text)])

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

wordlist = []
myFiles = os.listdir('Abstracts/')
for files in myFiles:
	infile = open(path+str(files))
	a = infile.read()
	a = strip_punctuation(a)
	lowera = to_lower(a) # makes lowercase
	tokens = word_tokenize(lowera) # makers into l
	tokens = [ps.stem(word) for word in tokens] # Root stem of word, comes out weird
	#lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in tokens])
	#filtered = ' '.join([w for w in tokens if len(w) > 2]) # attempt at filtering out words 2 or less ch
	filtereda = [w for w in tokens if not w in stop_words] # filters out stopwords
	wordlist = wordlist + filtereda
	
	infile.close()

largeString = FreqDist(wordlist)
largeString.plot(50)
	

with open("dictionarynltk.txt", 'w') as f:
	print(largeString, file=f)
#for key in sorted(wordlist.values()):
#	print("%s: %s" % (key,wordlist[key]))
#outfile.write(wordlist)
#outfile.close()

