#! /usr/bin/python

'''
This is a template outlining the functions we are expecting for us to be able to
interface with an call your code. This is not all of the functions you need. You
will also need to make sure to bring your decision tree learner in somehow
either by copying your code into a learn_decision_tree function or by importing
your decision tree code in from the file your wrote for PA#1. You will also need
some functions to handle creating your data bags, computing errors on your tree,
and handling the reweighting of data points.
For building your bags numpy's random module will be helpful.
'''

# This is the only non-native library to python you need
import numpy as np;
import sys, os;
import math;
import random;
from collections import OrderedDict
'''
Function: load_and_split_data(datapath)
datapath: (String) the location of the UCI mushroom data set directory in memory

This function loads the data set. datapath points to a directory holding
agaricuslepiotatest1.csv and agaricuslepiotatrain1.csv. The data from each file
is loaded and returned. All attribute values are nomimal. 30% of the data points
are missing a value for attribute 11 and instead have a value of "?". For the
purpose of these models, all attributes and data points are retained. The "?"
value is treated as its own attribute value.

Two nested lists are returned. The first list represents the training set and
the second list represents the test set.
'''
def load_data(datapath):
    train_fname = datapath + "/" + "agaricuslepiotatrain1.csv"
    test_fname = datapath + "/" + "agaricuslepiotatest1.csv"
    my_data_train = []
    with open(train_fname) as f:
        data = f.readlines()[1:]
        for ln in data:
            dat = []
            temp = ln.strip(" ").split(",")[0:21]
            temp = [int(i) for i in temp]
            temp1 = ln.strip(" ").split(",")[22:]
            temp1 = [int(i) for i in temp1]
            dat = temp[0:20] + temp1 + list(str(temp[20]))
            my_data_train.append(dat)

    my_data_test = []
    with open(test_fname) as f:
        data = f.readlines()[1:]
        for ln in data:
            dat = []
            temp = ln.strip(" ").split(",")[0:21]
            temp = [int(i) for i in temp]
            temp1 = ln.strip(" ").split(",")[22:]
            temp1 = [int(i) for i in temp1]
            dat = temp[0:20] + temp1 + list(str(temp[20]))
            my_data_test.append(dat)

    return my_data_train,my_data_test

def uniq_val(datapath):
    train_fname = datapath + "/" + "agaricuslepiotatrain1.csv"
    j = 0
    unq_dict={}
    with open(train_fname) as f:
        data = f.readline()
        feat_set = data.strip("\n").split(",")[0:20] + data.strip("\n").split(",")[22:]
    for i in feat_set:
        unq_dict[i+"_"+str(j)] = [0,1]
        j+=1
    return unq_dict

def check(ln,new_tree):
    for i in new_tree.keys():
        feat = i[0].split("_")[-1]
        val = i[1]
        if ln[int(feat)] == val:
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
    bag = []
    for ln in data:
        if depth!= 0:
            prediction = check(ln,new_tree)
        else:
            prediction = str(new_tree)
        prediction = '0' if prediction is None else prediction
        bag.append(prediction)
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
    return bag,float(crt)/float(count)

#Method used to get the majority class from a leaf node or when a nodes gain is zero
def get_class(data):
    temp = {}
    val = 0
    for i in ['0','1']:
        temp[i]=0
    for ln in data:
        temp[ln[len(ln)-1]]+=1
    for i in ['0','1']:
        if val < temp[i]:
            val = temp[i]
            k = i
    return k

#Main Method where the decision tree is developed. All other methods are
#further called from this method.
#Returns a decision tree for the depth mentioned
def buildtree(data,depth):
    curr_depth = -1
    best_infogain = 0.0

    for i in sorted(val_dict):
        entropy_s = entropy(data)
        entropy_val = 0.0
        feat = i.split("_")[-1]
        for j in val_dict[i]:
            t_set,n_set = split_data(data,int(feat),j)
            val = float(len(t_set))/float(len(data))
            entropy_val = (val)*entropy(t_set) + (1-val)*entropy(n_set)
            full_ent = entropy_s - entropy_val
            if best_infogain < full_ent and len(t_set)>0 and len(n_set)>0:
                best_infogain = full_ent
                best_feature = (i,j)
                best_split = [t_set,n_set]

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
        if ln[len(ln)-1] == "1":
            tp += 1
        elif ln[len(ln)-1] == "0":
            fp += 1
    avg_tp = tp/float(count)
    avg_fp = fp/float(count)
    if avg_tp == 0.0 or avg_fp == 0.0:
        ent = 0.0
    else:
        ent = -((avg_tp)*math.log((avg_tp),2))-((avg_fp)*math.log((avg_fp),2))
    return ent

def create_bag(train_data):
    bag_train = []
    bag_train_full = []
    count = len(train_data)
    for i in range(count/2):
        temp = random.choice(train_data)
        bag_train.append(temp)
    bag_train_full = bag_train + bag_train
    return bag_train_full


'''
Function: learn_bagged(tdepth, numbags, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numbags: (Integer)the number of bags to use to learn the trees
datapath: (String) the location in memory where the data set is stored

This function will manage coordinating the learning of the bagged ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''
def learn_bagged(tdepth, numbags, datapath,train_data,test_data,val_dict):
    if tdepth == 0:
        clas_lab = get_class(train_data)
        print "Majority Class is : " + clas_lab
        acc = classify(test_data,clas_lab,tdepth)
        print "Accuracy :",acc
    else:
        full_bag = []
        for i in range(numbags):
            train_data = create_bag(train_data)
            tree = buildtree(train_data,tdepth)
            key = tree.keys()[-1]
            tree = tree[tree.keys()[-1]]
            new_tree = {}
            new_tree[key] = tree
            print new_tree
            bag,acc = classify(test_data,new_tree,tdepth)
            print "Accuracy :",acc
            print "*"*50
            print "\t" + "Decision Tree for Depth : %d"%tdepth
            print "*"*50
            full_bag.append(bag)
    print len(full_bag[0])

'''
Function: learn_boosted(tdepth, numtrees, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numtrees: (Integer) the number of boosted trees to learn
datapath: (String) the location in memory where the data set is stored

This function wil manage coordinating the learning of the boosted ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''
def learn_boosted(tdepth, numtrees, datapath):
    pass;


if __name__ == "__main__":
    # The arguments to your file will be of the following form:
    # <ensemble_type> <tree_depth> <num_bags/trees> <data_set_path>
    # Ex. bag 3 10 mushrooms
    # Ex. boost 1 10 mushrooms

    # Get the ensemble type
    entype = sys.argv[1];
    # Get the depth of the trees
    tdepth = int(sys.argv[2]);
    # Get the number of bags or trees
    nummodels = int(sys.argv[3]);
    # Get the location of the data set
    datapath = sys.argv[4];

    # Check which type of ensemble is to be learned
    if entype == "bag":
        # Learned the bagged decision tree ensemble
        train_data,test_data = load_data(datapath) #Lists with training and test data
        val_dict = uniq_val(datapath) #Dictionary with values for each feature
        tree = OrderedDict()
        learn_bagged(tdepth, nummodels, datapath,train_data,test_data,val_dict);
    else:
        # Learned the boosted decision tree ensemble
        learn_boosted(tdepth, nummodels, datapath);
