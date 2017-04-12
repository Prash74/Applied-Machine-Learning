#!/usr/bin/env python

"""
decisiontree.py: A python program to build a decision tree on a given data
set to train a ML Model.

Args:
    filename: name of the training set
    depth: depth of the decision tree to build

Class Labels: -
class: 0,1

Features and their possible values: -
a1:    1, 2, 3
a2:    1, 2, 3
a3:    1, 2
a4:    1, 2, 3
a5:    1, 2, 3, 4
a6:    1, 2
"""

import sys
import math

__author__ = "Prashanth Balasubramani, Siddharth Jayashankar"

def uniq_val(data):
    unq_dict={}
    temp = {}
    for i in range(0,6):
        temp[i] = []
    for ln in data:
        for i in range(0,len(ln)-1):
            temp[i].append(ln[i])
    for i in range(0,6):
        unq_dict[i] = list(set(temp[i]))
    return unq_dict

def buildtree(data):
    best_infogain = 0.0
    best_feature = 0
    #Calculating Entropy for Each feature
    for i in range(0,6):
        entropy_s = entropy(data)
        print "Entropy of Feature",i
        print "is",entropy_s
        entropy_val = 0.0
        for j in val_dict[i]:
            entropy_val += entropy(data,i,j)
        full_ent = entropy_s - entropy_val
        print "Info Gain is",full_ent
        print "for Feature",i
        if best_infogain < full_ent:
            best_infogain = full_ent
            best_feature = i
    print "Best Feature: ",best_feature

def split_data(data,col,value):
    set1 = []
    set2 = []
    for ln in data:
        if ln[col] >= value:
            set1.append(ln)
        else:
            set2.append(ln)
    return set1,set2

def entropy(data,feature=None,value=None):
    count = len(data)
    tp,fp = 0,0
    if value == None and feature == None:
        for ln in data:
            if ln[6] == "1":
                tp += 1
            elif ln[6] == "0":
                fp += 1
    else:
        for ln in data:
            if ln[feature] == value and ln[6] == "1":
                tp += 1
            elif ln[feature] == value and ln[6] == "0":
                fp += 1
    print tp,fp,count
    avg_tp = tp/float(count)
    avg_fp = fp/float(count)
    if avg_tp == 0.0 or avg_fp == 0.0:
        ent = 0.0
    else:
        ent = -((avg_tp)*math.log((avg_tp),2))-((avg_fp)*math.log((avg_fp),2))
    return ent

my_data = []
fname = sys.argv[1]
depth = sys.argv[2]

print "Input FileName: ",fname
print "Depth of Decision Tree:",depth

#Converting the input file to csv format
with open(fname) as f:
    data = f.readlines()
    for ln in data:
        temp = ln.strip(" ").split(" ")[0:7]
        temp = [int(i) for i in temp]
        dat = temp[1:7] + list(str(temp[0]))
        my_data.append(dat)

val_dict = uniq_val(my_data)
print val_dict
buildtree(my_data)
