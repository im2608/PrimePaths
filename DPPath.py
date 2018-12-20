'''
Created on Dec 19, 2018

@author: Heng.Zhang
'''

import numpy as np
import math 

class DPPath(object):
    def __init__(self, X, Y):
        self.N = len(X)
        self.dist = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(self.N):
                self.dist[i,j] = math.sqrt(pow(X[i]-X[j], 2) + pow(Y[i] - Y[j], 2))
                
        self.path = np.ones((2**(self.N+1),self.N))
        
        self.dp = np.ones((2**(self.N + 1), self.N))*-1

        return
    
    def TSP(self, s,init,num):
        if (self.dp[s][init] != -1):            
            return self.dp[s][init]

        if s==(1<<(self.N)):
            return self.dist[0][init]
        
        sumpath=1000000000
        for i in range(self.N):
            if s&(1<<i):
                m=self.TSP(s&(~(1<<i)),i,num+1)
                m += self.dist[i][init]
                if m<sumpath:
                    sumpath=m
                    self.path[s][init]=i
        self.dp[s][init]=sumpath
        return self.dp[s][init]    
    