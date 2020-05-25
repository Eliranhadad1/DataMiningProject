

def myNgrams(file_name):
   file_hiv = open(file_name, "r")
   ngrams_size = 4
   ngrams = {}
    
   #Start with the values that we need
   unuse_line= file_hiv.readline()

   while True:
     getFromFile = file_hiv.read(ngrams_size)
     
     if getFromFile not in ngrams.keys():
         ngrams[getFromFile] = []
     ngrams[getFromFile].append(file_hiv.tell())
     if len(getFromFile) < ngrams_size:
         break
   file_hiv.close()
   return ngrams
    

ngrams_hiv = myNgrams("hiv.fasta")
ngrams_sc = myNgrams("sc.fasta")



