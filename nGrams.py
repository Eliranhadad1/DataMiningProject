

def myNgrams(file_hiv):
   
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
     
   return ngrams
    

file_hiv = open("hiv.fasta", "r")
file_sc = open("sc.fasta", "r")

ngrams_hiv = myNgrams(file_hiv)
ngrams_sc = myNgrams(file_sc)

file_hiv.close()
file_sc.close()


