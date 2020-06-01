"""
N grams - divide text file to parts 

    input: text file name
           number of chars to read from file
        
    output: list of different parts  
    
"""
def nGrams(file_name,ngrams_size):
   
   file_reader=without_line_breaks(file_name)
   
   ngrams = []
   
   i=0
   while True:
     getFromFile = file_reader[i:i+ngrams_size]
     i+=ngrams_size
     if len(getFromFile) < ngrams_size:
         break
     if getFromFile not in ngrams: #without duplic..?
      ngrams.append(getFromFile)
        
   return ngrams   
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

# ngrams_hiv = nGrams("hiv.fasta",10)
# ngrams_sc = nGrams("sc.fasta",10)

