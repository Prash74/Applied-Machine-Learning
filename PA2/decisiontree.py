#!/usr/bin/env python

"""
decisiontree.py: A python program to build a decision tree on a given data
set to train a ML Model.

Args:
    Training_Filename: name of the training set
    depth            : depth of the decision tree to build
    Test_Filename    : name of the test set [Optional Parameter]

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
from collections import OrderedDict

__author__ = "Prashanth Balasubramani, Siddharth Jayasankar"

#Method used to display the Tree
def display_tree(tree,count=0):
    if isinstance(tree,dict):
        for k in tree.keys():
            print "\t"*count + str(k)
            count += len(tree.keys())
            print "\t"*count + "left:"
            display_tree(tree[k]['left'],count)
            print "\t"*count + "right:"
            display_tree(tree[k]['right'],count)
    else:
        print "\t"*count + tree

#Method used traverse the tree for each test data point and assign a prediction based
#on the tree built. Returns a class label based on our prediction
def check(ln,new_tree):
    for i in new_tree.keys():
        feat = int(i[0][1])-1
        val = i[1]
        if ln[feat] == val:
            mv = 'right'
        else:
            mv = 'left'
    if new_tree[i][mv] == '1':
        return '1'
    elif new_tree[i][mv] == '0':
        return '0'
    else:
        val = check(ln,new_tree[i][mv])
        return val

#Method used to run our ML model on the test data set and to calculate our
#decision tree's accuracy and the confusion matrix
def classify(data,new_tree,depth):
    tp,tn,fp,fn = 0,0,0,0
    count = len(data)
    crt = 0
    for ln in data:
        if depth!= 0:
            prediction = check(ln,new_tree)
        else:
            prediction = str(new_tree)
        prediction = '0' if prediction is None else prediction
        if prediction == '1' and int(ln[len(ln)-1]) == 1:
            tp += 1
            crt += 1
        elif prediction == '0' and int(ln[len(ln)-1]) == 0:
            tn += 1
            crt += 1
        elif prediction == '1' and int(ln[len(ln)-1]) == 0:
            fp += 1
        elif prediction == '0' and int(ln[len(ln)-1]) == 1:
            fn += 1
    print "Total Count :",count
    print "Correctly Classified :",crt
    print "Misclassified Count :",count-crt
    print "\t\tCONFUSION MATRIX"
    print "\t Predicted- \t\t Predicted+"
    print "Actual-   %d \t\t\t %d"%(tn,fp)
    print "Actual+   %d \t\t\t %d"%(fn,tp)
    return float(crt)/float(count)

#Method used to get the majority class from a leaf node or when a nodes gain is zero
def get_class(data):
    temp = {}
    val = 0
    for i in ['0','1']:
        temp[i]=0
    for ln in data:
        temp[ln[6]]+=1
    for i in ['0','1']:
        if val < temp[i]:
            val = temp[i]
            k = i
    return k

#Method used to pre-process the data sets(Training and Test) to get all unique values
#of each feature in the data set
def uniq_val(data):
    unq_dict={}
    temp = {}
    for i in ['a1','a2','a3','a4','a5','a6']:
        temp[i] = []
    for ln in data[1:]:
        temp['a1'].append(ln[0])
        temp['a2'].append(ln[1])
        temp['a3'].append(ln[2])
        temp['a4'].append(ln[3])
        temp['a5'].append(ln[4])
        temp['a6'].append(ln[5])
    for key in sorted(temp):
        unq_dict[key] = list(set(temp[key]))
    return unq_dict

#Main Method where the decision tree is developed. All other methods are further
#called from this method.
#Returns a decision tree for the depth mentioned
def buildtree(data,depth):
    curr_depth = -1
    best_infogain = 0.0
    feat = 0

    for i in sorted(val_dict):
        entropy_s = entropy(data)
        entropy_val = 0.0
        for j in val_dict[i]:
            t_set,n_set = split_data(data,feat,j)
            val = float(len(t_set))/float(len(data))
            entropy_val = (val)*entropy(t_set) + (1-val)*entropy(n_set)
            full_ent = entropy_s - entropy_val
            if best_infogain < full_ent and len(t_set)>0 and len(n_set)>0:
                best_infogain = full_ent
                best_feature = (i,j)
                best_split = [t_set,n_set]
        feat += 1

    depth -= 1
    if best_infogain > 0 and curr_depth != depth:
        tb = buildtree(best_split[0],depth)
        fb = buildtree(best_split[1],depth)
        tree[best_feature] = { "left":fb, "right":tb }
        return {best_feature:tree[best_feature]}
    else:
        return str(get_class(data))

#Method used to split a data set based on the given feature and the threshold value mentioned
def split_data(data,col,value):
    set1 = []
    set2 = []
    for ln in data:
        if ln[col] == value:
            set1.append(ln)
        else:
            set2.append(ln)
    return set1,set2

#Method used to calculate entropy of a given data split_data
#Return a entropy value
def entropy(data):
    count = len(data)
    if count == 0:
        ent = 0.0
        return ent
    tp,fp = 0,0
    for ln in data:
        if ln[6] == "1":
            tp += 1
        elif ln[6] == "0":
            fp += 1
    avg_tp = tp/float(count)
    avg_fp = fp/float(count)
    if avg_tp == 0.0 or avg_fp == 0.0:
        ent = 0.0
    else:
        ent = -((avg_tp)*math.log((avg_tp),2))-((avg_fp)*math.log((avg_fp),2))
    return ent

my_data = [] #To store the training data set
train_fname = sys.argv[1] #Training Data set File Name
depth = int(sys.argv[2]) #Depth of the decision tree being developed

print "Input FileName: ",train_fname
print "Depth of Decision Tree:",depth

#Pre-processing the training data set to our desired format
with open(train_fname) as f:
    data = f.readlines()
    for ln in data:
        temp = ln.strip(" ").split(" ")[0:7]
        temp = [int(i) for i in temp]
        dat = temp[1:7] + list(str(temp[0]))
        my_data.append(dat)

#Building a dictionary with all the features and its possible values
val_dict = uniq_val(my_data)

tree = OrderedDict() #Our Decision tree being used

#if depth is zero : we can just get the majority class from the training dataset
#without any decision tree being developed
if depth == 0:
    clas_lab = get_class(my_data)
    print "Majority Class is : " + clas_lab

#Otherwise build a decision tree on the given depth
else:
    buildtree(my_data,depth)
    key = tree.keys()[-1]
    tree = tree[tree.keys()[-1]]
    new_tree = {}
    new_tree[key] = tree

#This step will only happen if we pass the test data set as an arguement
#Same Pre-processing happens with the test data set and classify() method is
#called which will return the Accuracy and display the confusion matrix for
#the test data set
if len(sys.argv)>3:
    my_data1 = []
    test_fname = sys.argv[3]
    print "Test File is: ",test_fname
    with open(test_fname) as f:
        data = f.readlines()
        for ln in data:
            temp = ln.strip(" ").split(" ")[0:7]
            temp = [int(i) for i in temp]
            dat = temp[1:7] + list(str(temp[0]))
            my_data1.append(dat)
    if depth != 0:
        acc = classify(my_data1,new_tree,depth)
        print "Accuracy :",acc
        print "*"*50
        print "\t" + "Decision Tree for Depth : %d"%depth
        print "*"*50
        display_tree(new_tree)
    else:
        acc = classify(my_data1,clas_lab,depth)
        print "Accuracy :",acc

    #To display the developed Decision Tree
