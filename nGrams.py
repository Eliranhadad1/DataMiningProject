"""
N grams - divide text file to parts 

    input: text file name
           number of chars to read from file
        
    output: list of different parts  
    
"""
from nltk import ngrams
import matplotlib.pyplot as plt

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
def nGrams(file_name,ngrams_size):
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
    
    #show Histogram
     
    plt.hist(list_hist,bins=len(freq))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('N GRAMS')
    plt.ylabel('Frequency')
    plt.title('Histogram n grams= '+str(ngrams_size))
    plt.show()
    
    return [ngrams_array,freq,list_hist] 









