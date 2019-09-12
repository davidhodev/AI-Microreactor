from pybliometrics.scopus import ScopusSearch
from pybliometrics.scopus import AbstractRetrieval
import pandas as pd
import os
import time
from tqdm import tqdm
import sys

os.system('cls' if os.name == 'nt' else 'clear')

def lookup():
	search = input('Enter Search Terms\n')
	option = input('Enter 1 for Exact search, 0 for inexact search\n')

	if option == '1':
		query = '{' + search + '}' # exact search
	else:
		query = 'TITLE-ABS-KEY( ' + search + ')' # inexact search

	s = ScopusSearch(query, download=False)

	print('Number of results: ')
	length = s.get_results_size()
	print(length)

	if length > 0:
		dl = input('Would you like to download the results y/n\n')
		if dl == 'y':
			s = ScopusSearch(query, download=True)
			dataframe = pd.DataFrame(pd.DataFrame(s.results)) # converts results into a dataframe
			pd.options.display.max_colwidth = 150
			pd.options.display.max_rows = None
			print(dataframe[['eid', 'title']])
			dataframe.iloc[:,0] = dataframe.iloc[:,0].astype(str) # converts the eid dataframe objects to string
			
			option2 = input('\n Enter the row of the abstract you want to download, or enter ALL to download all\n')
				
			if option2 == 'ALL':
				for i in progressbar(range(length), "Download Progress ", 40):
					ab = AbstractRetrieval(dataframe.iloc[i,0],view='FULL') # searches for abstracts using eid
					with open(os.path.join('/home/benjamin/Python_Codes/Abstracts',dataframe.iloc[i,0] + '.txt'), 'w') as f:
						f.write("%s\n" % ab.abstract) #creates individual txt files titled by their eid
			else:
				try:
					val = int(option2)
					print('Attempting to download abstract with eid ' + dataframe.iloc[val,0])
					ab = AbstractRetrieval(dataframe.iloc[val,0],view='FULL') # searches for abstracts using eid
					with open(os.path.join('/home/benjamin/Python_Codes/Abstracts',dataframe.iloc[val,0] + '.txt'), 'w') as f:
						f.write("%s\n" % ab.abstract)
					print('Success!\n')
				except ValueError:
					print('Invalid row number\n')
	else:
		print('No results found, please try again\n')

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()  				
		
running = True

while running == True:  #basic loop to reset the search
	lookup()
	ask = input('Run search again? y/n\n')
	if ask == 'y':
		continue
	else:
		running = False
