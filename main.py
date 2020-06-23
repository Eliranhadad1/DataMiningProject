# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 08:20:58 2020

@author: User
"""


import scipy.stats as ss
import numpy as np
from zv import zv
import matplotlib.pyplot as plt
from mykmedoids import KMedoids
from nltk import ngrams
import nltk

def calculateNGrams(D,minN,maxN):
   
   Dngrams=[]
   for j in range(len(D)):
     Dngrams.append([])
     for i in range(minN,maxN+1):
         string_tuples=ngrams(D[j],i)
         fdist = nltk.FreqDist(string_tuples)
         f=0
         for [k,z] in fdist.items():
              f+=1

         Dngrams[j].append(f)
    
   return Dngrams   
def calculateRank(Di):
    # rankArray=len(Di)-ss.rankdata(Di,method='min').astype(int)
    rankArray=ss.rankdata(Di,method='min').astype(int)
    return np.array(rankArray).tolist()
def column(matrix, i):
    return [row[i] for row in matrix]
def showGraph(x,y,xlabel,ylabel,string_title):
    # df_fitbit_activity = pd.DataFrame({'x': x, 'y': y})
    # df_fitbit_activity.set_index('x')['y'].plot();
    # sns.set(font_scale=1.4)
    # df_fitbit_activity.set_index('x')['y'].plot(figsize=(12, 10), linewidth=2.5, color='maroon')
    plt.plot(x, y) 
    
    plt.xlabel(xlabel, labelpad=15)
    plt.ylabel(ylabel, labelpad=15)
    plt.title(string_title, y=1.02, fontsize=22);
    plt.show()
def getRankByD(D):
    
    ranks=[]
    for j in range(len(D)):
      ranks.append(calculateRank(D[j]))
      
    return ranks
def calculateZv(rank_file):
    pi=[]
    deltaTi=doAppend(len(rank_file))
   
    for d in range(0,len(rank_file)):
        deltaTi[d]=rank_file[0:d]
    
   
    for r in range(len(rank_file)):
        pi.append(zv(rank_file[r],deltaTi[r]))
  
    
    return [pi,deltaTi]
def showZVGraphs(di_file,zv,fileName):
    #matih
    di_Index=[]
    # for ii in range(1,len(di_file)):
    for ii in range(len(di_file)):
        di_Index.append(ii)
    
    showGraph(di_Index, zv, "Di", "ZV", "ZV graph file "+fileName)
def calculateDZV(Di,Dj,Delta_i,Delta_j,zv_i,zv_j):
    if len(Delta_i)==0 or len(Delta_j)==0:
        return
    dzvResult = 0
    dzvResult+=zv_i
    dzvResult+=zv_j
    dzvResult-=zv(Di,Delta_j)
    dzvResult-=zv(Dj,Delta_i)
    
    return abs(dzvResult)
def calculateAllDZV(Di,Dj,Delta_i,Delta_j,zv1,zv2):

    dvz_matrix=[]
    #rows
    for ii in range((len(Di)+len(Dj))):
     dvz_matrix.append([])
     #columns
     for jj in range((len(Di)+len(Dj))):
      if ii<len(Di):
          if jj<len(Di):
              dvz_matrix[ii].append(calculateDZV(Di[ii], Di[jj], Delta_i[ii], Delta_i[jj],zv1[ii],zv2[jj]))
          else:
              jj2=jj-len(Di)
              dvz_matrix[ii].append(calculateDZV(Di[ii], Dj[jj2], Delta_i[ii], Delta_j[jj2],zv1[ii],zv2[jj2]))
      else:   
         ii2=ii-len(Di)
         if jj<len(Di):
              dvz_matrix[ii].append(calculateDZV(Dj[ii2], Di[jj], Delta_j[ii2], Delta_i[jj],zv1[ii2],zv2[jj]))
         else:
              jj2=jj-len(Di)
              dvz_matrix[ii].append(calculateDZV(Dj[ii2], Dj[jj2], Delta_j[ii2], Delta_j[jj2],zv1[ii2],zv2[jj2]))
    
    return dvz_matrix 
def doAppend( size=10000 ):
    result = []
    for i in range(size):
        message= "some unique object %d" % ( i, )
        result.append(message)
    return result
def showDZVGraph(dzv_matrix):
    #matih
    dzv_matrix_Index=[]
    for ii in range(len(dzv_matrix)):
        dzv_matrix_Index.append(ii)
    
    showGraph(dzv_matrix_Index, dzv_matrix, "D", "DZV", "DZV graph files ")
def preProcessing(fileName,m):
  fileReader=without_line_breaks(fileName)
  return DivideToMSubDocs(fileReader,m)
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
def DivideToMSubDocs(file,m):
    D=[]
  
    lenOfD =round(len(file)/m)
    for i in range(m):
      D.append(file[i*lenOfD:(i*lenOfD+lenOfD)])
      
    return D
def ShowHist(x,y):
    
    # x = ranks
    listForHist = []
    
    for ng in range(len(x)):
     for r in range(y[ng]):
            listForHist.append(ng+3)
    
    plt.hist(listForHist, bins=8, color='#0504aa',alpha=0.7, rwidth=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('N GRAMS')
    plt.ylabel('Ranks')
    plt.title('Histogram D')
    plt.show()
####################################


#Measurement of file 1
fileName="sc.fasta"
m=200
D = preProcessing(fileName, m)

# #n grams from 4-10
ngrams_D=calculateNGrams(D,3,10)
rank_file1 = getRankByD(ngrams_D)
for di in range(len(rank_file1)):
    ShowHist(ngrams_D[di],rank_file1[di])

[zv_file1,Delta_i]=calculateZv(rank_file1)

showZVGraphs(D,zv_file1,fileName)

#Measurement of file 2
fileName="hiv.fasta"

D = preProcessing(fileName, m)

# #n grams from 4-10
ngrams_D=calculateNGrams(D,3,10)
rank_file2 = getRankByD(ngrams_D)

for di in range(len(rank_file1)):
    ShowHist(ngrams_D[di],rank_file2[di])

[zv_file2,Delta_j]=calculateZv(rank_file2)

showZVGraphs(D,zv_file2,fileName)


#Measurement of DZV
dzv_matrix=calculateAllDZV(rank_file1,rank_file2,Delta_i,Delta_j,zv_file1,zv_file2)
showDZVGraph(dzv_matrix)

# Dividing DZV matrix
# until di in the rows
firstDZV = dzv_matrix[1:len(rank_file1)]
secondDZV = dzv_matrix[len(rank_file1)+1:]

y=dzv_matrix
y=np.delete(y,0,0)
y=np.delete(y,m-1,0)
y=np.delete(y,0,1)
y=np.delete(y,m-1,1)

X = y

kmedoids = KMedoids(n_clusters=2, random_state=0).fit(X)
l=kmedoids.labels_
c=kmedoids.cluster_centers_
d_index=[]
for jj in range(len(y)):
    d_index.append(jj)    
showGraph(d_index, l, 'Di', 'cluster', 'KMedoids K=2')
##########################

