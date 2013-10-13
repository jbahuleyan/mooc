# Learning from Data.
# Week#1. Homework#1
#
import random
import numpy
from numpy import array
from random import random

def RandPoint(l, u):
    return array([random()*(u-l)+l, random()*(u-l)+l])

def Classify(line, pt):
    return 1 if ((pt[0]-line[0][0])*(line[1][1]-line[0][1])-(line[1][0]-line[0][0])*(pt[1]-line[0][1])) > 0 else -1

def PlaClassify(pt, W):
    return 1 if W.dot(pt) > 0 else -1

def PlaTrain(line, samples):
    w = array([0, 0, 0])
    biasSamples = list(map(lambda s : array([1] + list(s)), samples))
    trueClass = list(map(lambda p : Classify(line, p), samples))
    learnt, N = False, 0
    while not learnt:
        plaClass = list(map(lambda p : PlaClassify(p, w), biasSamples))
        N += 1
        n = len(samples)
        i = int(random()*n)
        while n>0:
            if trueClass[i] != plaClass[i]:
                w = w + trueClass[i]*biasSamples[i]
                break
            n -= 1
            i = (i+1)%len(samples)
        if n==0:
            learnt = True
    return (N, w)

def PlaError(w, line, Points):
    trueClass = list(map(lambda p : Classify(line, p), Points))
    biasedPoints = list(map(lambda s : array([1] + list(s)), Points))
    plaClass = list(map(lambda p : PlaClassify(p, w), biasedPoints))
    return len(list(filter(lambda i : plaClass[i] != trueClass[i], range(len(Points)))))*1.0/len(Points)
