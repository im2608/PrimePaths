'''
Created on Dec 20, 2018

@author: Heng.Zhang
'''

from global_param import *
import pandas as pd    
import math
import numpy as np

# 代码参考 https://blog.csdn.net/yg838457845/article/details/81127697， 有修改

class Solution:
    def __init__(self,X,start_node):
        self.X = X #距离矩阵
        self.start_node = start_node #开始的节点
        self.array = [[0]*(2**len(self.X)) for i in range(len(self.X))] #记录处于x节点，未经历M个节点时，矩阵储存x的下一步是M中哪个节点
        self.dp = np.ones((len(self.X), 2**len(self.X)))*-1

    def transfer(self,sets):
        su = 0
        for s in sets:
            su = su + 2**s # 二进制转换
        return su
    
    # tsp总接口
    def tsp(self):
        s = self.start_node
        num = len(self.X)
        cities = list(range(num)) #形成节点的集合
        past_sets = [s] #已遍历节点集合
        cities.pop(cities.index(s)) #构建未经历节点的集合
        node = s #初始节点
        return self.solve(node,cities) #求解函数
    
    # 从 node 出发，遍历 future_set 中所有的节点 （node 不在 future_set 中）
    def solve(self,node,future_sets):
        su = self.transfer(future_sets)
        if (self.dp[node][su] != -1):
            return self.dp[node][su]
        
        # 迭代终止条件，表示没有了未遍历节点, 连接当前节点和起点或结束
        if len(future_sets) == 0:
            return 0
#             return self.X[node][self.start_node]
        
        d = 99999
        # node如果经过future_sets中节点，最后回到原点的距离
        distance = []
        # 遍历未经历的节点
        for i in range(len(future_sets)):
            s_i = future_sets[i]
            copy = future_sets[:]
            copy.pop(i) # 删除第i个节点，认为已经完成对其的访问
            d = self.solve(s_i,copy)
            distance.append(self.X[node][s_i] + d)
        # 动态规划递推方程，利用递归
        d = min(distance)
        # node需要连接的下一个节点
        next_one = future_sets[distance.index(d)]
        # 回溯矩阵，（当前节点，未遍历节点集合）——>下一个节点
        self.array[node][su] = next_one
        self.dp[node][su] = d
        return d
    
    def get_path(self):
        path = [self.start_node]
        lists = list(range(len(self.X)))
        start = self.start_node
        while len(lists) > 0:
            lists.pop(lists.index(start))
            m = self.transfer(lists)
            next_node = self.array[start][m]
            path.append(next_node)
            start = next_node
            
        return path
    
if (__name__ == '__main__'):    
    dataframe = pd.read_csv("%s/../data/cities.csv" % runningPath)
    
    v = dataframe.iloc[0:20,1:3]
     
    train_v= np.array(v)
    train_d=train_v
    D = np.zeros((train_v.shape[0],train_d.shape[0]))
     
    #计算距离矩阵
    for i in range(train_v.shape[0]):
        for j in range(train_d.shape[0]):
            D[i,j] = math.sqrt(np.sum((train_v[i,:]-train_d[j,:])**2))
            D[i,j] = math.sqrt(np.sum((train_v[i,:]-train_d[j,:])**2))
    
    s = time.clock()
    S = Solution(D,0)
    print(S.tsp())
    print(S.get_path())
    
    # 开始回溯
    # lists = list(range(len(S.X)))
    # start = S.start_node
    # while len(lists) > 0:
    #     lists.pop(lists.index(start))
    #     m = S.transfer(lists)
    #     next_node = S.array[start][m]
    #     print (start,"--->" ,next_node)
    #     start = next_node
    
    e = time.clock()
    print("running time %s"%(e-s))