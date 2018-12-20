'''
Created on Dec 11, 2018

@author: Heng.Zhang
'''
import pandas as pd
import csv
import matplotlib.pyplot as plt
import math

def calculate_city_distance():

    cities_coordinate = csv.reader(open(r'F:\doc\ML\Kaggle\PrimePaths\data\cities.csv', 'r'))
    
    index = 0;
    X = []
    Y = []
    prime_cities = []
    for each_city in cities_coordinate:
        if (index == 0):
            index += 1
            continue

        id = int(each_city[0])
        X.append(float(each_city[1]))
        Y.append(float(each_city[1]))
        index += 1
        if (index % 1000 == 0):
            print("%d read\r" % index, end='')
            
        if (isPrimeNum(id)):
            prime_cities.append(id)
            
    primeCities = pd.DataFrame(prime_cities)
    primeCities.to_csv(r'F:\doc\ML\Kaggle\PrimePaths\data\Primecities.csv', header=None, index=False)
    
    distance_file = open(r'F:\doc\ML\Kaggle\PrimePaths\data\city_distance.csv', 'w')
    
    for i in range(0, len(X) - 2):
        distance_i_j = []
        for j in range(i + 1, len(X) - 1):
            dist = round(math.sqrt(pow(X[i] - X[j], 2) + pow(Y[i] - Y[j], 2)), 4)
            distance_i_j.append(str(dist))
            if (j % 100 == 0):
                print("city %d distance %d calculated\r" % (i, j), end='')
        
        distance_file.write("%s\n" % ",".join(distance_i_j))
#     drawCities(X, Y)

    return


def drawCities(X, Y):
    plt.scatter(X, Y)
    plt.show()

def isPrimeNum(n):
    if n <= 1:
        return False
    elif n % 2 == 0 and n != 2:     # 去除能被2整除的  不包括2（其实也可以包括2，以为2没有素数对）
        return False
    elif n % 3 == 0 and n != 3:     # 去除能被3整除的  不包括3
        return False
    elif n % 5 == 0 and n != 5:     # 去除能被5整除的  不包括5
        return False
    elif n % 7 == 0 and n != 7:     # 去除能被7整除的  不包括7
        return False
    else:
        for i in range(3, int(math.sqrt(n)) + 1, 2):   # 这里 +1是将开方后的结果包含在内
                if n % i == 0:
                    return False
        return True



calculate_city_distance()