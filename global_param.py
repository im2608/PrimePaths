'''
Created on Dec 13, 2018

@author: Heng.Zhang
'''
import time
import sys

runningPath = sys.path[0]

def getCurrentTime():
    return "[%s]" % (time.strftime("%Y-%m-%d %X", time.localtime()))
