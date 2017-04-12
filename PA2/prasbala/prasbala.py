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
from collections import Counter
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

# A method to display the decision tree.
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

def load_data(datapath,boost=None):
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
            if boost != None:
                if temp[20] == 1:
                    temp[20] = "+1"
                elif temp[20] == 0:
                    temp[20] = "-1"
            dat = temp[0:20] + temp1 + [str(temp[20])]
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
            if boost != None:
                if temp[20] == 1:
                    temp[20] = "+1"
                elif temp[20] == 0:
                    temp[20] = "-1"
            if boost != None:
                dat = [0] + temp[0:20] + temp1 + [str(temp[20])]
            else:
                dat = temp[0:20] + temp1 + list(str(temp[20]))
            my_data_test.append(dat)

    return my_data_train,my_data_test

#Method used to pre-process the data sets(Training and Test) to get all unique values
#of each feature in the data set
def uniq_val(datapath,weights=None):
    train_fname = datapath + "/" + "agaricuslepiotatrain1.csv"
    if weights == None:
        j = 0
    else:
        j = 1
    unq_dict={}
    with open(train_fname) as f:
        data = f.readline()
        feat_set = data.strip("\n").split(",")[0:20] + data.strip("\n").split(",")[22:]
    for i in feat_set:
        unq_dict[i+"_"+str(j)] = [0,1]
        j+=1
    return unq_dict

#Method used traverse the tree for each test data point and assign a prediction based
#on the tree built. Returns a class label based on our prediction
def check(ln,new_tree):
    for i in new_tree.keys():
        feat = i[0].split("_")[-1]
        val = i[1]
        if ln[int(feat)] == val:
            mv = 'right'
        else:
            mv = 'left'
    if new_tree[i][mv] == '1' or new_tree[i][mv] == '+1':
        return new_tree[i][mv]
    elif new_tree[i][mv] == '0' or new_tree[i][mv] == '-1':
        return new_tree[i][mv]
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
    return bag,float(crt)/float(count)

#Method used to get the majority class from a leaf node or when a nodes gain is zero
def get_class(data,boost=None):
    if boost == None:
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
    else:
        temp = {}
        val = 0
        for i in ['-1','+1']:
            temp[i]=0
        for ln in data:
            temp[ln[len(ln)-1]]+=1
        for i in ['-1','+1']:
            if val < temp[i]:
                val = temp[i]
                k = i
    return k

#Main Method where the decision tree is developed. All other methods are further
#called from this method.
#Returns a decision tree for the depth mentioned
def buildtree(data,depth,weights=None):
    if weights == None:
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
    else:
        curr_depth = -1
        best_infogain = 0.0

        for i in sorted(val_dict):
            entropy_s = entropy(data,weights)
            entropy_val = 0.0
            feat = i.split("_")[-1]
            wt_full = 0.0
            for k in range(len(data)):
                wt_full += data[k][0]
            for j in val_dict[i]:
                t_set,n_set = split_data(data,int(feat),j)
                tp_wt,fp_wt = 0.0,0.0
                for ln in t_set:
                    tp_wt += ln[0]
                for ln in n_set:
                    fp_wt += ln[0]
                avgtp_wt = tp_wt/float(wt_full)
                avgfp_wt = fp_wt/float(wt_full)
                entropy_val = (avgtp_wt)*entropy(t_set,weights) + (avgfp_wt)*entropy(n_set,weights)
                full_ent = entropy_s - entropy_val
                if best_infogain < full_ent and len(t_set)>0 and len(n_set)>0:
                    best_infogain = full_ent
                    best_feature = (i,j)
                    best_split = [t_set,n_set]

        depth -= 1
        if best_infogain > 0 and curr_depth != depth:
            tb = buildtree(best_split[0],depth,"boost")
            fb = buildtree(best_split[1],depth,"boost")
            tree[best_feature] = { "left":fb, "right":tb }
            return {best_feature:tree[best_feature]}
        else:
            return str(get_class(data,"boost"))

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
def entropy(data,weights=None):
    count = 0
    if weights == None:
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
    else:
        count = len(data)
        wt_full = 0.0
        for i in range(len(data)):
            wt_full += data[i][0]
        if count == 0:
            ent = 0.0
            return ent
        tp,fp,tp_wt,fp_wt = 0,0,0.0,0.0
        for ln in data:
            if ln[len(ln)-1] == '+1':
                tp += 1
                tp_wt += ln[0]
            elif ln[len(ln)-1] == '-1':
                fp += 1
                fp_wt += ln[0]
        avg_tp = tp/float(count)
        avg_fp = fp/float(count)
        avgtp_wt = tp_wt/float(wt_full)
        avgfp_wt = fp_wt/float(wt_full)
        if avg_tp == 0.0 or avg_fp == 0.0:
            ent = 0.0
        else:
            ent = -((avgtp_wt)*math.log((avgtp_wt),2))-((avgfp_wt)*math.log((avgfp_wt),2))
        return ent

#Method used to sub-sample the training data during bagging.
#Uses python random() function to select train data samples
def create_bag(train_data):
    bag_train = []
    bag_train_full = []
    count = len(train_data)
    for i in range(count/2):
        temp = random.choice(train_data)
        bag_train.append(temp)
    bag_train_full = bag_train + bag_train
    return bag_train_full

#Method used to calculate and predict for test data using the
#bags produced by our bagging model
def bag_accuracy(bag,test_data):
    tp,tn,fp,fn = 0,0,0,0
    count = len(test_data)
    crt = 0
    idx = 0
    for ln in test_data:
        prediction = bag[idx]
        idx+=1
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
    print "Accuracy :",float(crt)/float(count)

#Method used to calculate and predict for test data using the
#different trees produced by our boosting model
def boost_accuracy(data,new_tree,alpha,test=None):
    if test == None:
        tp,tn,fp,fn = 0,0,0,0
        count = len(data)
        crt = 0
        tot_wt = 0
        for ln in data:
            tot_wt += ln[0]
            prediction = check(ln,new_tree)
            if prediction == '+1' and ln[len(ln)-1] == '+1':
                tp += 1
                crt += 1
            elif prediction == '-1' and ln[len(ln)-1] == '-1':
                tn += 1
                crt += 1
            elif prediction == '+1' and ln[len(ln)-1] == '-1':
                fp += 1
            elif prediction == '-1' and ln[len(ln)-1] == '+1':
                fn += 1
        err = float(count-crt)/float(count)
        alpha = (1/float(2))*math.log(((1-err)/float(err)),2)
        c1,c2,c3 = 0,0,0
        for ln in data:
            prediction = check(ln,new_tree)
            ln[0] = (ln[0] * math.exp(-1*alpha*int(prediction)*int(ln[len(ln)-1])))/float(tot_wt)
        return alpha,data
    else:
        tp,tn,fp,fn = 0,0,0,0
        count = len(data)
        crt = 0
        temp = []
        for ln in data:
            prediction = check(ln,new_tree)
            temp.append(alpha*int(prediction))
            if prediction == '+1' and ln[len(ln)-1] == '+1':
                tp += 1
                crt += 1
            elif prediction == '-1' and ln[len(ln)-1] == '-1':
                tn += 1
                crt += 1
            elif prediction == '+1' and ln[len(ln)-1] == '-1':
                fp += 1
            elif prediction == '-1' and ln[len(ln)-1] == '+1':
                fn += 1
        return temp

#Method used for prediction of test data using the summarized
#values of each tree produced thru boosting ML model
def boost_classify(boost_bag,test_data):
    i,k=0,1
    count = len(test_data)
    crt = 0
    if tdepth == 1:
        k = 2
    tp,tn,fp,fn = 0,0,0,0
    for ln in test_data:
        for j in range(k):
            temp = boost_bag[j][i]
        i+=1
        if temp < 0:
            prediction = '-1'
        else:
            prediction = '+1'
        if prediction == '+1' and ln[len(ln)-1] == '+1':
            tp += 1
            crt += 1
        elif prediction == '-1' and ln[len(ln)-1] == '-1':
            tn += 1
            crt += 1
        elif prediction == '+1' and ln[len(ln)-1] == '-1':
            fp += 1
        elif prediction == '-1' and ln[len(ln)-1] == '+1':
            fn += 1
    print "Total Count :",count
    print "Correctly Classified :",crt
    print "Misclassified Count :",count-crt
    print "\t\tCONFUSION MATRIX"
    print "\t Predicted- \t\t Predicted+"
    print "Actual-   %d \t\t\t %d"%(tn,fp)
    print "Actual+   %d \t\t\t %d"%(fn,tp)
    print "Accuracy :",float(crt)/float(count)


'''
Function: learn_bagged(tdepth, numbags, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numbags: (Integer)the number of bags to use to learn the trees
datapath: (String) the location in memory where the data set is stored

This function will manage coordinating the learning of the bagged ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''
def learn_bagged(tdepth,numbags,datapath,train_data,test_data,val_dict):
    print "*"*50
    print "\t" + "Algorithm : Bagging"
    print "\t" + "Decision Tree for Depth : %d"%tdepth
    print "\t" + "Number of Bags : %d"%numbags
    print "*"*50
    if tdepth == 0:
        clas_lab = get_class(train_data)
        print "Majority Class is : " + clas_lab
        a,acc = classify(test_data,clas_lab,tdepth)
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
            bag,acc = classify(test_data,new_tree,tdepth)
            full_bag.append(bag)
        final_bag = []
        for i in range(len(full_bag[0])):
            temp = []
            for j in range(numbags):
                temp.append(full_bag[j][i])
            t = Counter(temp)
            final_bag.append(t.most_common(1)[0][0])
        bag_accuracy(final_bag,test_data)

'''
Function: learn_boosted(tdepth, numtrees, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numtrees: (Integer) the number of boosted trees to learn
datapath: (String) the location in memory where the data set is stored

This function wil manage coordinating the learning of the boosted ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''
def learn_boosted(tdepth,numtrees,datapath,train_data,test_data,val_dict):
    print "*"*50
    print "\t" + "Algorithm : Boosting"
    print "\t" + "Decision Tree for Depth : %d"%tdepth
    print "\t" + "Number of Trees : %d"%numtrees
    print "*"*50
    alpha = 0
    for i in range(len(train_data)):
        train_data[i] = [1/float(len(train_data))] + train_data[i]
    boost_train = train_data
    boost_test = []
    a = []
    for i in range(numtrees):
        tree = {}
        tree = buildtree(boost_train,tdepth,"boost")
        key = tree.keys()[-1]
        tree = tree[tree.keys()[-1]]
        new_tree = {}
        new_tree[key] = tree
        alpha,boost_train = boost_accuracy(boost_train,new_tree,alpha)
        boost_test.append(boost_accuracy(test_data,new_tree,alpha,"boost"))
    boost_classify(boost_test,test_data)

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

    if entype == "bag":
        val_dict = uniq_val(datapath) #Dictionary with values for each feature
    else:
        val_dict = uniq_val(datapath,"boost")
    tree = OrderedDict()
    # Check which type of ensemble is to be learned
    if entype == "bag":
        train_data,test_data = load_data(datapath) #Lists with training and test data
        # Learned the bagged decision tree ensemble
        learn_bagged(tdepth,nummodels,datapath,train_data,test_data,val_dict);
    else:
        train_data,test_data = load_data(datapath,"boost") #Lists with training and test data
        #print train_data
        # Learned the boosted decision tree ensemble
        learn_boosted(tdepth,nummodels,datapath,train_data,test_data,val_dict);
