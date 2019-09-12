from mat2vec.processing import MaterialsTextProcessor
from collections import OrderedDict
import operator
import os
import string
text_processor = MaterialsTextProcessor()
path = "/home/benjamin/Python_Codes/Abstracts/"
savepath = "/home/benjamin/Python_Codes/Dictionary/"


wordlist = {}
myFiles = os.listdir('Abstracts/')
for files in myFiles:
	infile = open(path+str(files))
	a = infile.read()
	a= a.translate(str.maketrans("", "", string.punctuation))

	[words,chemicals] = text_processor.process(a)
	
	for word in words:
		if word in wordlist:		
			wordlist[word] += 1
		else:
			wordlist[word] = 1
	infile.close()

outfile = open("dictionary.txt",'w')
sorteddict = sorted(wordlist.items(), key=operator.itemgetter(1))
print(sorteddict)
largeString = ""
for key,val in sorteddict:
	largeString += key + ": "
	largeString += str(val)
	largeString += "\n"

with open("dictionary2.txt", 'w') as f:
	print(largeString, file=f)
#for key in sorted(wordlist.values()):
#	print("%s: %s" % (key,wordlist[key]))
#outfile.write(wordlist)
#outfile.close()

