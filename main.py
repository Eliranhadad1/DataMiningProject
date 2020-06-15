# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 08:20:58 2020

@author: User
"""

# import nGrams
from nGrams import nGrams
from scipy.stats import spearmanr
import scipy.stats as ss
import numpy as np
from zv import zv

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def calculateNGrams(fileName):
    D = []
    for i in range(3, 4):
        x=nGrams(fileName, i)  
        D.append(x[1])
    
    return D
    
def calculateRank(Di):
    rankArray=len(Di)-ss.rankdata(Di,method='min').astype(int)
    return np.array(rankArray).tolist()

def column(matrix, i):
    return [row[i] for row in matrix]
def showGraph(x,y,xlabel,ylabel,string_title):
    df_fitbit_activity = pd.DataFrame({'x': x, 'y': y})
    df_fitbit_activity.set_index('x')['y'].plot();
    sns.set(font_scale=1.4)
    df_fitbit_activity.set_index('x')['y'].plot(figsize=(12, 10), linewidth=2.5, color='maroon')
    plt.xlabel(xlabel, labelpad=15)
    plt.ylabel(ylabel, labelpad=15)
    plt.title(string_title, y=1.02, fontsize=22);
    plt.show()
def getRankByD(D):
    
    maxLen=len(D[len(D)-1])
    
    # #add 0 in short lists then maxLen 
    for jj in range(len(D)):
        for ii in range(maxLen-len(D[jj])):
            D[jj].append(0)
    #transporm for keep di in the same row
    Di=[]
    for j in range(maxLen):
        Di.append(column(D, j))
    
    #calcute range of Di
    ranks=[]
    for j in range(len(Di)):
      ranks.append(calculateRank(Di[j]))
      
    # newDi=[]
    # for g in range(len(Di)):
    #  newDi.append(Di[g])
      
    return [ranks,Di]
def calculateZv(rank_file):
    pi=[]
    deltaTi=[]
   
    for d in range(1,len(rank_file)):
     deltaTi.append(0)
     deltaTi[d-1]=rank_file[0:d]
    
        
    for r in range(1,len(rank_file)):
        pi.append(zv(rank_file[r],deltaTi[r-1]))
    
    return [pi,deltaTi]
def showZVGraphs(di_file,zv,fileName):
    #matih
    di_Index=[]
    for ii in range(len(di_file)):
        di_Index.append(ii)
    
    showGraph(di_Index, zv, "Di", "ZV", "ZV graph file "+fileName)
 
def calculateDZV(Di,Dj,Delta_i,Delta_j):
    dzvResult = 0
    for i in range(len(Di)):
        dzvResult+=zv(Di,Delta_i)
        dzvResult+=zv(Dj,Delta_j)
        dzvResult-=zv(Di,Delta_j)
        dzvResult-=zv(Dj,Delta_i)
    return dzvResult
def calculateAllDZV(Di,Dj,Delta_i,Delta_j):
    
    dvz_matrix=[]
    for ii in range(1,len(Di)):
     dvz_matrix.append(0)
     for jj in range(1,len(Dj)):
        dvz_matrix[ii-1].append(calculateDZV(Di[ii], Dj[jj], Delta_i[ii-1], Delta_j[jj-1]))
    
    return dvz_matrix 

####################################

#Measurement of file 1
fileName="sc.fasta"
#n grams from 4-6
D = calculateNGrams(fileName)
[rank_file1,di_file1] = getRankByD(D)
#Delta_i - list of all the delta
# for example :
# Delta_i[0] == Delta2 (d1)
# Delta_i[2] == Delta4 (d1,d2,d3)
[zv_file1,Delta_i]=calculateZv(rank_file1)
[rank_file1,Delta_i,zv_file1]=showZVGraphs(di_file1,zv_file1,fileName)

#Measurement of file 2
fileName="hiv.fasta"
#n grams from 4-6
D2 = calculateNGrams(fileName)
[rank_file2,di_file2] = getRankByD(D2)
[zv_file2,Delta_j]=calculateZv(rank_file2)
[rank_file2,Delta_j,zv_file2]=showZVGraphs(di_file2,zv_file2,fileName)

#Measurement of DZV
dzv_matrix=calculateAllDZV(rank_file1,rank_file2,Delta_i,Delta_j)



##########################
# from pyclustering.cluster.kmedoids import kmedoids
# https://stackoverflow.com/questions/56765715/k-medoids-in-python-pyclustering



