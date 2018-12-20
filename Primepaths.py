'''
Created on Dec 12, 2018

@author: Heng.Zhang
'''
import csv
import numpy as np
import math
import logging
import datetime

from global_param import *
from TSP2 import *


class Cities(object):
    def __init__(self, max_city_aounds, around_x_len, around_y_len):
        
        print("%s starting, max city arounds %d, around x %d, around %d" % (getCurrentTime(), max_city_aounds, around_x_len, around_y_len))

        cities_coordinate = csv.reader(open(r'%s/../data/cities.csv' % runningPath, 'r'))

        self.X = []
        self.Y = []
        
        index = 0;
        for each_city in cities_coordinate:
            if (index == 0):
                index += 1
                continue
    
            id = int(each_city[0])
            self.X.append(float(each_city[1]))
            self.Y.append(float(each_city[2]))
            index += 1
            if (index % 1000 == 0):
                print("%d read\r" % index, end='')

        self.sorted_X = np.argsort(self.X)
        self.sorted_Y = np.argsort(self.Y)
        self.grid_size = 24
        self.max_x = 5100
        self.max_y = 3400
        
        self.right_x = self.max_x // self.grid_size
        self.top_y = self.max_y // self.grid_size
        self.grid0_x = self.X[0] // self.grid_size
        self.grid0_y = self.Y[0] // self.grid_size

        self.prime_cities = set()       
        prime_cities_csv = csv.reader(open(r'%s/../data/Primecities.csv' % runningPath, 'r'))
        for each_prime in prime_cities_csv:
            self.prime_cities.add(int(each_prime[0]))
            
        self.cities_passed_through = set([0])
        
        self.max_around_cities = max_city_aounds
        self.around_x_len = around_x_len
        self.around_y_len = around_y_len
        
        log_file = r'%s/../log/cities_%d_%s.log' % (runningPath, self.grid_size, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=log_file,
                            filemode='w')
        
        print("grid size %d, right x %d , grid0 x %d, top y %d, grid 0 y%d, expected grids %d" % 
              (self.grid_size, self.right_x, self.grid0_x, self.top_y, self.grid0_y, self.right_x * self.top_y))

    
    def get_grid_count(self):
        for grid_size in range(10, 100, 1):
            self.right_x = self.max_x // grid_size
            self.top_y = 3400 // grid_size
            self.grid0_x = self.X[0] // grid_size
            self.grid0_y = self.Y[0] // grid_size
            
            if ((self.right_x - 1) % 2 != 0 and  (self.grid0_x % 2) != 0 and 
                (self.top_y - 1) % 2 != (self.grid0_y % 2)):
                print("=== candiate: grid size %d, (right x - 1, grid0 x) %d, %d, (top y - 1, grid 0 y) %d, %d" % 
                      (grid_size, self.right_x - 1, self.grid0_x, self.top_y - 1, self.grid0_y))
#             else:
#                 print("grid size %d, (right x - 1, grid0 x) %d, %d, (top y - 1, grid 0 y) %d, %d" % 
#                       (grid_size, self.right_x - 1, self.grid0_x, self.top_y - 1, self.grid0_y))
        return
    
    def get_cities_around_on_axis(self, city_idx, coordinate_axis, sorted_coordinate_axis, around_len):            
        cities_around_axis = set()
        cities_around_cnt = 0

        for sorted_idx in range(len(sorted_coordinate_axis)):
            if (sorted_coordinate_axis[sorted_idx] == city_idx):
                break

        around_idx = 1
        while (True):
            dist_left_or_bottom = around_len + 1
            if (sorted_idx - around_idx >= 0):
                dist_left_or_bottom = coordinate_axis[sorted_coordinate_axis[sorted_idx]] - coordinate_axis[sorted_coordinate_axis[sorted_idx - around_idx]] 
                if (dist_left_or_bottom <= around_len and 
                    sorted_coordinate_axis[sorted_idx - around_idx] not in self.cities_passed_through):
                    cities_around_axis.add(sorted_coordinate_axis[sorted_idx - around_idx])

            dist_right_or_top = around_len + 1
            if (sorted_idx + around_idx < len(coordinate_axis)):
                dist_right_or_top = coordinate_axis[sorted_coordinate_axis[sorted_idx + around_idx]] - coordinate_axis[sorted_coordinate_axis[sorted_idx]] 
                if (dist_right_or_top <= around_len and 
                    sorted_coordinate_axis[sorted_idx + around_idx] not in self.cities_passed_through):
                    cities_around_axis.add(sorted_coordinate_axis[sorted_idx + around_idx])

            if (cities_around_cnt == len(cities_around_axis) and 
                dist_left_or_bottom > around_len and 
                dist_right_or_top > around_len):
                break

            cities_around_cnt = len(cities_around_axis)
            around_idx += 1

        return cities_around_axis;
    
    def get_cities_around(self, city_idx, around_x_len, around_y_len):
        cities_around = set()
        while (len(cities_around) < self.max_around_cities):
            cities_around_x = self.get_cities_around_on_axis(city_idx, self.X, self.sorted_X, around_x_len)
            while (len(cities_around_x) == 0):
                around_x_len *= 2
                cities_around_x = self.get_cities_around_on_axis(city_idx, self.X, self.sorted_X, around_x_len)
    
            cities_around_y = self.get_cities_around_on_axis(city_idx, self.Y, self.sorted_Y, around_y_len)
            while (len(cities_around_y) == 0):
                around_y_len *= 2
                cities_around_y = self.get_cities_around_on_axis(city_idx, self.Y, self.sorted_Y, around_y_len)
            
            cities_around = cities_around_x & cities_around_y
            if (len(cities_around) > self.max_around_cities):
                cities_around = set(list(cities_around)[0 : self.max_around_cities])
                
            around_x_len *= 2
            around_y_len *= 2
        
#         print("%s city %d has %d cities around\r" % (getCurrentTime(), city_idx, len(cities_around)), end='')
        return cities_around
    
 
    
    def cities_dist(self, c1, c2):
        return round(math.sqrt(pow(self.X[c1] - self.X[c2], 2) + pow(self.Y[c1] - self.Y[c2], 2)), 4)
    
    def search_path_dp(self, move_dist, city_move_path, cities_around_set, shortest_grid_move_dist):
        if (len(cities_around_set) == 0):
            return move_dist, city_move_path
        
        if ( move_dist >= shortest_grid_move_dist):
            return None, None
        
        shortest_move_dist = 10000000
        shortest_city_move_path = city_move_path
        
        for each_city in cities_around_set:
            dist_of_next_move = move_dist + self.cities_dist(city_move_path[-1], each_city)
            path_of_next_move = city_move_path + [each_city]
            candidate_cities_set = cities_around_set.difference(set([each_city]))
            next_move_dist, next_city_move_path = self.search_path_dp(dist_of_next_move, path_of_next_move, candidate_cities_set, shortest_grid_move_dist)
            if (next_move_dist is not None and next_city_move_path is not None and 
                next_move_dist < shortest_move_dist):
                shortest_move_dist = next_move_dist 
                shortest_city_move_path = next_city_move_path
                
        return shortest_move_dist, shortest_city_move_path
    
    def verify_path(self, distance, move_path):
        dist = 0;
        for i in range(0, len(move_path) - 1):
            dist += self.cities_dist(move_path[i], move_path[i+1])
            
        return dist == distance


    def find_path(self):
        move_file = open(r'%s/../output/city_move_%d_%d_%d.csv' % (runningPath, self.max_around_cities, self.around_x_len, self.around_y_len), 'w')
        move_file.write("Path\n")
        
        current_city = 0
        shortest_city_move = [0]
        city_move_list = [0, []]

        while (len(self.cities_passed_through) < len(self.X)):
            shortest_move_dist = 10000000
            cities_around_set = self.get_cities_around(current_city, self.around_x_len, self.around_y_len)

            for each_city in cities_around_set:
                next_move_dist = self.cities_dist(current_city, each_city)
                city_move_path = [current_city, each_city]
                candidate_cities_set = cities_around_set.difference(set([each_city]))
                move_dist, city_move_path = self.search_path_dp(next_move_dist, city_move_path, candidate_cities_set, shortest_move_dist)
                
                if (not self.verify_path(move_dist, city_move_path)):
                    print("%s failed verify %d, %f, %s" % (getCurrentTime(), current_city, move_dist, city_move_path))
                    exit(-1)
                
                if (move_dist < shortest_move_dist):
                    shortest_move_dist = move_dist
                    shortest_city_move = city_move_path

            self.cities_passed_through = self.cities_passed_through.union(set(shortest_city_move))
            current_city = shortest_city_move[-1]
            city_move_list[0] += shortest_move_dist
            city_move_list[1].extend(shortest_city_move[1:])

            print("%s %d cities have passed through, distance %f\r" % (getCurrentTime(), len(self.cities_passed_through), city_move_list[0]), end='')
            if (len(self.cities_passed_through) % 2000 == 0):
                print("%s %d cities went, distance %f" % (getCurrentTime(), len(self.cities_passed_through), city_move_list[0]))
                
            logging.info("%s %d cities have passed through, distance %f" % (getCurrentTime(), len(self.cities_passed_through), city_move_list[0]))
        
        for each_city in city_move_list[1]:
            move_file.write("%d\n" % each_city)
           
        return
    
    def get_cities_in_grid_bin(self, city_axis, sorted_city, left, right, grid_axis):
        if (left >= right):
            return left

        if (city_axis[sorted_city[left]] < grid_axis and 
            city_axis[sorted_city[left + 1]] >= grid_axis):
            return left + 1
        
        mid = (left + right) // 2 
        grid_mid = city_axis[sorted_city[mid]]
        if (grid_mid == grid_axis):
            return mid
        
        if (grid_axis < grid_mid):
            return self.get_cities_in_grid_bin(city_axis, sorted_city, left, mid-1, grid_axis)
        else:
            return self.get_cities_in_grid_bin(city_axis, sorted_city, mid+1, right, grid_axis)
        
    def get_cities_in_grid(self, grid_left, grid_bottom):
        cities_around_x = set()
        cities_around_y = set()
        
        first_city_idx = self.get_cities_in_grid_bin(self.X, self.sorted_X, 0, len(self.sorted_X) - 1, grid_left)
        for idx in range(first_city_idx, len(self.sorted_X)):
            if (self.X[self.sorted_X[idx]] > grid_left + self.grid_size):
                break
            cities_around_x.add(self.sorted_X[idx])
            
        first_city_idx = self.get_cities_in_grid_bin(self.Y, self.sorted_Y, 0, len(self.sorted_Y) - 1, grid_bottom)
        for idx in range(first_city_idx, len(self.sorted_Y)):
            if (self.Y[self.sorted_Y[idx]] > grid_bottom + self.grid_size):
                break
            cities_around_y.add(self.sorted_Y[idx])

        cities_around = cities_around_x & cities_around_y
        if (0 in cities_around):
            cities_around.remove(0)

        return list(cities_around)
    
    def find_path_with_grid(self):
        
        move_file = open(r'%s/../output/city_grid_move_%d_%s.csv' % (runningPath, self.grid_size, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), 'w')
        move_file.write("Path\n0\n")
        
        grid_path_x, grid_path_y = self.get_grid_path()
        
        print("%s total grids %d" % (getCurrentTime(), len(grid_path_x)))

        shortest_city_move = [0]
        city_move_list = [0, []]

        current_city = 0
        
        for grid_idx in range(0, len(grid_path_x)):
            grid_left = grid_path_x[grid_idx]* self.grid_size
            grid_bottom = grid_path_y[grid_idx]* self.grid_size
            cities_in_grid_list = self.get_cities_in_grid(grid_left, grid_bottom)
            if (len(cities_in_grid_list) == 0):
                continue
            
            N = len(cities_in_grid_list)
            print("%s grid %d has %d cities\r" % (getCurrentTime(), grid_idx, N), end='')
            logging.info("%d,%d" % (grid_idx, N))
            continue
            
            cities_x_in_grid = [self.X[current_city]]
            cities_y_in_grid = [self.Y[current_city]]
            
            cities_x_in_grid.extend([self.X[city] for city in cities_in_grid_list])
            cities_y_in_grid.extend([self.Y[city] for city in cities_in_grid_list])
            
            D = np.zeros((len(cities_x_in_grid), len(cities_x_in_grid)))
             
            #计算距离矩阵
            for i in range(N + 1):
                for j in range(N + 1):
                    D[i,j] = math.sqrt(pow(cities_x_in_grid[i]-cities_x_in_grid[j], 2) + pow(cities_y_in_grid[i]-cities_y_in_grid[j], 2))
            
            dp = Solution(D, 0)
            shortest_move_dist = dp.tsp()
            path = dp.get_path()
            shortest_city_move = [cities_in_grid_list[i - 1] for i in path[1:-1]]

            self.cities_passed_through = self.cities_passed_through.union(set(shortest_city_move))
            current_city = shortest_city_move[-1]
            city_move_list[0] += shortest_move_dist
            city_move_list[1].extend(shortest_city_move)

            print("%s %d cities %d grids went, distance %f\r" % (getCurrentTime(), len(self.cities_passed_through), grid_idx, city_move_list[0]), end='')
            if (len(self.cities_passed_through) % 2000 == 0):
                print("%s %d cities %d grids went, distance %f" % (getCurrentTime(), len(self.cities_passed_through), grid_idx, city_move_list[0]))
                
            logging.info("%s %d cities have passed through, distance %f" % (getCurrentTime(), len(self.cities_passed_through), city_move_list[0]))
        
        for each_city in city_move_list[1]:
            move_file.write("%d\n" % each_city)
        
        move_file.write("0\n")
        return

    def verify_submition(self, filename):
        move_file_csv = csv.reader(open(r'%s/../output/%s' % (runningPath, filename), 'r'))
        move_cities = []
        move_X = []
        move_Y = []
        index = 0
        for each_move in move_file_csv:
            if (index == 0):
                index += 1
                continue

            city = int(each_move[0])
            move_cities.append(city)
            move_X.append(self.X[city])
            move_Y.append(self.Y[city])

            index += 1
            if (index % 1000 == 0):
                print("%d read\r" % index, end='')

        verify_total_dist = 0
        no_prime_number_cnt = 0
        for i in range(len(move_cities) - 1):
            cur_city = move_cities[i]
            next_city = move_cities[i+1]
            if (i % 10 == 0):
                if (cur_city not in self.prime_cities):
                    no_prime_number_cnt += 1

            verify_total_dist += self.cities_dist(cur_city, next_city)
            if (i % 1000 == 0):
                print("%d passd through\r" % i, end='')

        print("verify_total_dist %f, no_prime_number_cnt %d" % (verify_total_dist, no_prime_number_cnt))
        
        self.draw_move(move_X, move_Y)
        
    def draw_move(self, move_X, move_Y):
        import matplotlib.pyplot as plt
        from matplotlib import collections  as mc
        
        lines = [[(move_X[i], move_Y[i]),(move_X[i+1], move_Y[i+1])] for i in range(0,len(move_X)-2)]
        lc = mc.LineCollection(lines, linewidths=1)
        fig, ax = plt.subplots(figsize=(20,20))
        ax.set_aspect('equal')
        ax.add_collection(lc)
        ax.autoscale()

            
    def get_grid_path(self):
        grid_x = self.grid0_x
        grid_y = self.grid0_y

        grid_path_x = []
        grid_path_y = []

        # y0 以上的部分，从下到上，左右来回遍历
        while (grid_y < self.top_y):            
            while (grid_x < self.right_x):
                grid_path_x.append(grid_x)
                grid_path_y.append(grid_y)
                grid_x += 1
                
            grid_y += 1
            if (grid_y >= self.top_y):
                break
                
            grid_x = self.right_x - 1

            while (grid_x > 0):                
                grid_path_x.append(grid_x)
                grid_path_y.append(grid_y)
                grid_x -= 1

            grid_y += 1
            grid_x = 1

        grid_y = self.top_y - 1
        grid_x = 0

        while (grid_y >= self.grid0_y ):
            
            grid_path_x.append(grid_x)
            grid_path_y.append(grid_y)
            grid_y -= 1
        
        grid_y = self.grid0_y
        grid_x = 1

        while (grid_x < self.grid0_x):            
            grid_path_x.append(grid_x)
            grid_path_y.append(grid_y)
            grid_x += 1

        print("y0 above finished, grid_path len ", len(grid_path_x))

        # y0 以下的部分， 从左到右，上下来回遍历
        grid_y = self.grid0_y - 1
        while (grid_x < self.right_x):
            while (grid_y >= 1):
                grid_path_x.append(grid_x)
                grid_path_y.append(grid_y)
                grid_y -= 1

            grid_x += 1
            tmp = int(grid_x)
            if (tmp == self.right_x): # 用 if (grid_x == self.right_x) 会抛出异常， why？
                break

            grid_y = 1
            while (grid_y < self.grid0_y):                
                grid_path_x.append(grid_x)
                grid_path_y.append(grid_y)
                grid_y += 1

            grid_x += 1
            grid_y = self.grid0_y - 1
        
        grid_x = self.right_x - 1
        while (grid_x >= 0):            
            grid_path_x.append(grid_x)
            grid_path_y.append(grid_y)
            grid_x -= 1

        grid_x = 0
        grid_y = 1
        while (grid_x < self.grid0_x):
            while (grid_y < self.grid0_y):
                grid_path_x.append(grid_x)
                grid_path_y.append(grid_y)
                grid_y += 1

            grid_x += 1
            if (grid_x == self.grid0_x):
                break

            grid_y = self.grid0_y - 1
            while (grid_y > 0):
                grid_path_x.append(grid_x)
                grid_path_y.append(grid_y)
                grid_y -= 1
                
            grid_x += 1
            grid_y = 1
            
        print("grid_path len ", len(grid_path_x))

        return grid_path_x, grid_path_y
        
def main():
    max_city_aounds = int(sys.argv[1])
    around_x_len = int(sys.argv[2])
    around_y_len = int(sys.argv[3])
    cities = Cities(max_city_aounds, around_x_len, around_y_len)
#     cities.get_grid_count()
    cities.find_path_with_grid()
#     cities.find_path()    
    
#     cities.draw_move(grid_path_x, grid_path_y)
#     cities.verify_submition("city_grid_move_10.csv")
    
    return

if __name__ == '__main__':
    main()
    
    
