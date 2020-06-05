"""
N grams - divide text file to parts 

    input: text file name
           number of chars to read from file
        
    output: list of different parts  
    
"""
import nltk
from nltk import ngrams
import matplotlib.pyplot as plt
import numpy as np

def nGrams(file_name,ngrams_size):
   
   file_reader=without_line_breaks(file_name)
   
   ngrams = []
   freq = []
   
   i=0
   while True:
     getFromFile = file_reader[i:i+ngrams_size]
     i+=ngrams_size
     if len(getFromFile) < ngrams_size:
         break
     if getFromFile not in ngrams: #without duplic..?
      ngrams.append(getFromFile)
      freq.append(1)
    
     else:
         freq[ngrams.index(getFromFile)]=freq[ngrams.index(getFromFile)]+1
     
        
   return [ngrams,freq]   
def without_line_breaks(file_name):
   
 a_file = open(file_name, "r")
 #Start with the values that we need
 a_file.readline()
 
 string_without_line_breaks = ""
 for line in a_file:
    stripped_line = line.rstrip()
    string_without_line_breaks += stripped_line
        
 a_file.close()
 return string_without_line_breaks

#matih
# move one letter at the time
def newNGrams(file_name,ngrams_size):
    file_reader=without_line_breaks(file_name)
    
    list_hist =[]
    freq = []
    ngrams_array = []
    string_tuples = ngrams(file_reader,ngrams_size)
    
    for s in string_tuples:
     s1=""
     for j in range(len(s)):
      s1+=s[j]
     if s1 not in ngrams_array:
      ngrams_array.append(s1)
      freq.append(1)
     else:
      freq[ngrams_array.index(s1)]=freq[ngrams_array.index(s1)]+1
     list_hist.append(s1)
    
    return [ngrams_array,freq,list_hist] 

f = []
ng = [1,2,3,4,5,6,7,8,9,10]
for i in range(0,10):
    [ngrams_hiv,f1] = nGrams("hiv.fasta",ng[i])
    f.append(f1)
# [ngrams_hiv,f1] = nGrams("hiv.fasta",ng)
# [ngrams_sc,f2] = nGrams("sc.fasta",ng)

# plt.hist( f1,ngrams_hiv)
# plt.show()

ngNew=4
x = newNGrams("hiv.fasta",ngNew)


plt.hist(x[2],bins=len(x[1]))
# [n,bins,patches]=plt.hist(x[1], bins=arrayrange,histtype='stepfilled', color=['b'],alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('N GRAMS')
plt.ylabel('Frequency')
plt.title('Mati Histogram n grams= '+str(ngNew))
# plt.text(23, 45, r'$\mu=15, b=3$')
plt.show()
# maxfreq = n.max()
# Set a clean upper y-axis limit.








