# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 08:20:58 2020

@author: User
"""


import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
from mykmedoids import KMedoids
from nltk import ngrams 
import nltk
from fuzzywuzzy import fuzz
import operator
from scipy.stats import spearmanr

def zv(Di,deltaT):
    
    T=len(deltaT)
    if T==0:
        return 
    
    sigma=0;
    for Dj in deltaT:
         di=Di
         toAdd = len(di)-len(Dj)
         if toAdd>0:
          maxDj=max(Dj)
          for jj in range(toAdd):
              Dj.append(maxDj+jj+1)
              
         elif toAdd<0:
          maxDi=max(di)
          for jj in range(abs(toAdd)):
              di.append(maxDi+jj+1)
              
         [c,p]=spearmanr(di,Dj)
         sigma+=c
    return (1/ T)*sigma

def calculateNGrams(D,n,tolerancePrecent):
    
   
   Dngrams=[]
   
   for j in range(len(D)):
    Dngrams.append(0)
    Dngrams[j]={}
    string_tuples=ngrams(D[j],n)
    fdist = nltk.FreqDist(string_tuples)
    
    for [key,freq] in fdist.items():
        # key = ''.join(key)
        # if key=="CCAACT":
        #     print('x')
        fuzzyList={}
        for k in Dngrams[j].keys():
            fuzr = fuzz.ratio(key,k)
            if  fuzr >= tolerancePrecent:
                fuzzyList[k]=fuzr

        if  len(fuzzyList)== 0:
            Dngrams[j][key]=freq
        else:
            maxKey=max(fuzzyList.items(), key=operator.itemgetter(1))[0]
            Dngrams[j][maxKey]=Dngrams[j][maxKey]+freq
    
                
   DfrenkValues=[]
   for d in Dngrams:
        DfrenkValues.append(list(d.values()))
    
   return Dngrams,DfrenkValues

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
def calculateZv(D):
    pi=[]
    deltaTi=doAppend(len(D))
   
    for d in range(len(D)):
        deltaTi[d]=D[0:d]
    
   
    for r in range(len(D)):
        pi.append(zv(D[r],deltaTi[r]))
  
    
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
    listForHist=y
    # for ng in range(len(x)):
    #  for r in range(y[ng]):
    #         listForHist.append(ng+3)
    
    # x = np.arange(len(y))
    plt.bar(x, height=y)
    # plt.xticks(x, ['a','b','c'])
    
    # plt.hist(listForHist, bins=len(y), density=True, color='#0504aa',alpha=0.7, rwidth=0.2)
    # plt.grid(axis='y', alpha=0.75)
    plt.xlabel('N GRAMS Keys')
    plt.ylabel('Freq')
    plt.title('Histogram D')
    plt.show()
def makeClusters(X,k):
    kmedoids = KMedoids(n_clusters=k, random_state=0).fit(X)
    l=kmedoids.labels_

    d_index=[]
    for jj in range(len(X)):
        d_index.append(jj)    
    showGraph(d_index, l, 'Di', 'cluster', 'KMedoids K=2')
    return l
####################################
def calculateTheFilesData(n):
    
    #Measurement of file 1
    fileName="sc.fasta"
    m=100
    D = preProcessing(fileName, m)
    
    # #n grams from 4-10
    ngrams_D,DfrenqValues1=calculateNGrams(D,n,85)
    
    # for di in range(m):
    #     ShowHist(range(len(ngrams_D[di].keys())),list(ngrams_D[di].values()))
    
    
    [zv_file1,Delta_i]=calculateZv(DfrenqValues1)
    
    showZVGraphs(D,zv_file1,fileName)
    
    #Measurement of file 2
    fileName="hiv.fasta"
    
    D = preProcessing(fileName, m)
    
    # #n grams from 4-10
    ngrams_D,DfrenqValues2=calculateNGrams(D,n,85)
    
    for di in range(m):
        ShowHist(range(len(ngrams_D[di].keys())),list(ngrams_D[di].values()))
    
    
    [zv_file2,Delta_j]=calculateZv(DfrenqValues2)
    
    showZVGraphs(D,zv_file2,fileName)
    
    
    #Measurement of DZV
    dzv_matrix=calculateAllDZV(DfrenqValues1,DfrenqValues2,Delta_i,Delta_j,zv_file1,zv_file2)
    showDZVGraph(dzv_matrix)
    
    # Dividing DZV matrix
    # until di in the rows
    firstDZV = dzv_matrix[1:len(DfrenqValues1)]
    secondDZV = dzv_matrix[len(DfrenqValues2)+1:]
    
    y=dzv_matrix
    y=np.delete(y,0,0)
    y=np.delete(y,m-1,0)
    y=np.delete(y,0,1)
    y=np.delete(y,m-1,1)
    
    l=makeClusters(y,2)
    
    label_1 = 1
    label_2 = 0
    
    errorCount=0
    flag=0
    for hh in range(len(l)):
        if errorCount>=4:
            flag=1
        if hh < m/2-1:
          if l[hh]==label_2:
            errorCount+=1
          else:
              errorCount=0
        else:
            if l[hh]==label_2:
                 errorCount+=1
            else:
              errorCount=0
    if flag==0:
        print("Corona is not transgenic!")
    else:
        print("Corona is transgenic!")
    
    retMatrix=[]
    for row in firstDZV:
        retMatrix.append(row[:100])
    
    return retMatrix 
##########################

onlyCoronaDZV = calculateTheFilesData()

makeClusters(onlyCoronaDZV,2)
