import numpy as np
from math import log
import csv
import sys
import random
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def get_entropy(data_set):
    labels_count , record_nums = {} , len(data_set)
    z = 0
    while z < len(data_set):
        data = data_set[z]
        label = data[-1]
        labels_count[label] = labels_count.get(label, 0) + 1
        z+=1
    entropy = 0.0
    for key in labels_count:
        p = float(labels_count[key])/record_nums
        entropy -= p*log(p,2)
    return  entropy

def partition_kfold(set_complete, block):
    begin, finish = len(set_complete)/5 * block, len(set_complete)/5 * (block+1) - 1
    set1, set2 = list(), list()
    i = 0
    while i < len(set_complete):
        if i>=begin and i<=finish :
            set2.append(set_complete[i])
        else:
            set1.append(set_complete[i])
        i+=1
    return set1, set2


def get_conditional_entropy(branch_dict, total_record_nums):
    entropy = 0.0
    for feature_value in branch_dict.keys():
        p = float(len(branch_dict[feature_value]))/total_record_nums
        entropy += p*get_entropy(branch_dict[feature_value])
    return entropy


def create_tree(date_sets, feature_labels):
    labels = [list_v[-1] for list_v in date_sets]
    labels_set = set(labels)
    if len(labels_set) == 1:
        return labels_set.pop()
    feature_num = len(date_sets[0])-1
    if feature_num == 1:
        return get_most_common_value(labels)
    entropy_oringal = get_entropy(date_sets)
    record_nums , information_gain , best_feature_index = len(date_sets) , 0 , 0.0
    information_gain = 0.0
    best_information_gain = 0.0
    best_brahcn_dict = {}
    for index in range(0,feature_num):
        branch_dict = get_branch_sets(date_sets,index)
        entropy_condition = get_conditional_entropy(branch_dict,record_nums)
        information_gain = entropy_oringal - entropy_condition
        if(information_gain>best_information_gain):
            best_information_gain = information_gain
            best_feature_index = index
            best_brahcn_dict = branch_dict
    if 0 == information_gain:
        return get_most_common_value(labels)
    feature = feature_labels[best_feature_index]
    del feature_labels[best_feature_index]
    tree_node = {}
    for feature_value in best_brahcn_dict.keys():
        tree_node[feature_value] = create_tree(best_brahcn_dict[feature_value], feature_labels[:])
    tree = {}
    tree[feature] = tree_node
    return tree

def search_tree(tree, feature_labels, test_data):
    feature = tree.keys()[0]
    feature_index = feature_labels.index(feature)
    sub_tree = tree[feature][test_data[feature_index]]
    if isinstance(sub_tree, dict):
        return search_tree(sub_tree, feature_labels, test_data)
    else :
        return sub_tree

def get_most_common_value(labels_list):
    count_dict = {}
    for label in labels_list:
        count_dict[label] = count_dict.get(label,0) + 1
    keys = count_dict.keys()
    temp = keys[0]
    for key in keys:
        if count_dict[key] > count_dict[temp]:
            temp = key
    return temp

def get_branch_sets(data_set, feature_index):
    branch_sets = []
    branch_dict = {}
    for record in data_set:
        value = record[feature_index]
        if value not in branch_dict:
            branch_dict[value] = []
        new_record = record[0:feature_index]
        new_record.extend(record[feature_index+1:])
        branch_dict[value].append(new_record)
    for key in branch_dict.keys():
        branch_sets.append(branch_dict[key])
    return branch_dict

def classify_bench(train_set, feature_labels, test_data_set):
    '''
        Classification Benchmark.
    '''
    result = list()
    tree = create_tree(train_set, feature_labels[:])
    for test_data in test_data_set:
        result.append(search_tree(tree, \
            feature_labels, test_data))
    return result

def discretize_dataset(data_set, splite_value_dict):
    '''
        Discretize Dataset
    '''
    num_features = len(data_set[0]) - 1
    for i in range(num_features):
        for j in range(len(data_set)):
            if(data_set[j][i]<splite_value_dict[i]):
                data_set[j][i] = 0
            else:
                data_set[j][i] = 1

def split_data_set(data_set, feature_index, value):
    '''
        Splitting of Data, based on the feature index value.
    '''

    smaller_set , bigger_set = list() , list()
    
    for data in data_set:
        new_record = data[:feature_index]
        new_record.extend(data[feature_index+1:])
        if data[feature_index] >= value:
            bigger_set.append(new_record)
        else:
            smaller_set.append(new_record)
    
    return smaller_set,bigger_set

def normalize_data(setarray):
    '''
        Data Normalization.
    '''
    maxi = list()
    mini = list()
    for j in xrange(len(setarray[0])-1):
        maxi.append(setarray[0][j])
        mini.append(setarray[0][j])
    for i in xrange(len(setarray)):
        for j in xrange(len(setarray[i])-1):
            maxi[j] = max(maxi[j] , setarray[i][j])
            mini[j] = min(mini[j] , setarray[i][j])
    for i in xrange(len(setarray)):
        for j in xrange(len(setarray[i])-1):
            if maxi[j]!= mini[j]:
                setarray[i][j] = (setarray[i][j] - mini[j])/(maxi[j]-mini[j])
            else:
                setarray[i][j] = 1
    return setarray



def func(test_data, test_class, train_data, train_class):
    global param_grid
    dlf = GridSearchCV(SVC(kernel='rbf'), param_grid)
    dlf = dlf.fit(train_data, train_class)
    predicted = dlf.predict(test_data)
    print test_class , predicted
    return 1 - np.sum((test_class)^(predicted))*1.0/len(predicted)


def get_discretize_split_value_dict(data_set, clip_count=2):
    '''
        Calculate best-split value for each feature.
    '''
    best_split_value_dict = {}
    features_num = len(data_set[0]) - 1
    record_num = len(data_set)
    entropy_original = get_entropy(data_set)

    for index in range(0, features_num):
        #print "Feature no. ", str(index)
        best_information_gain = 0.0
        values = [record[index] for record in data_set]
        values_list =  list(set(values))
        values_num = len(values_list) 
        distance  = values_num/clip_count
        if clip_count > values_num:
            distance, clip_count = 1, values_num

        split_index = -1
        for _ in range(0, clip_count):
            split_index += distance
            splite_value = values_list[split_index]
            set1, set2 = split_data_set(data_set, index, splite_value)
            entropy_conditional = get_conditional_entropy(\
                {"set1":set1, "set2":set2}, record_num)
            infomation_gain = entropy_original - entropy_conditional
            if infomation_gain > best_information_gain:
                best_information_gain = infomation_gain
                best_split_value_dict[index] = splite_value

    return best_split_value_dict


def load_data(path):
    data_set = []
    #labels = ['sepal len' , 'sepal wid' , 'petal len' , 'petal wid']
    labels = list()
    
    for i in xrange(30):
        labels.append(i)
    
    csvfile = file(path,'rb')
    reader = csv.reader(csvfile)

    index = 0

    #first = True
    for s in reader:
        line = s
        
        #if first:
        #    labels.append(line)
        #    first=False
        #    continue

        label = line[-1]
        line = line[:-1]
        line =  map(float,line)
        line.append(label)
        data_set.append(line)
        index = index+1
    return data_set, labels


def run(file_path):
    '''
        Run Each Iteration.
    '''

    data_list, feature_labels = load_data(file_path)
    random.shuffle(data_list)

    train_size = 0.8

    total_size = len(data_list)
    train_num = int(train_size * total_size)

    ans = list()
    for i in xrange(train_num , total_size):
        ans.append(data_list[i][-1])

    splite_value_dict = get_discretize_split_value_dict(data_list[:train_num], 20)

    discretize_dataset(data_list,splite_value_dict)
    result = classify_bench(data_list[:train_num], feature_labels, data_list[train_num:])
    
    print classification_report(ans, result)
    print confusion_matrix(ans, result)

    total = 0
    correct = 0
    for i in xrange(len(ans)):
        if ans[i] == result[i]:
            correct+=1
        total+=1
    return float(correct)/total


def main(file_path):
    '''
        Main Code
    '''

    ans_list = list()
    for i in xrange(10):
        print '----Iter: ', str(i+1), '----' 
        val = run(file_path)
        ans_list.append(val)
    print ans_list
    
    ans_list = np.array(ans_list)

    print "Mean Value".ljust(10), ans_list.mean()
    print "Minimum Value".ljust(10), np.min(ans_list)
    print "Maximum Value".ljust(10), np.max(ans_list)

if __name__ == '__main__':
    main(sys.argv[1])