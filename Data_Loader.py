import os
from os import listdir
from os import walk

from os.path import isfile, join, splitext, isdir
from scipy.signal import butter, lfilter 
import numpy as np

def dense_to_one_hot(labels_dense, num_classes=5):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

def Data(onehot_,test_subnum):   
    subject_id = []
    # Used_Subject_id = []
    # Used_Subject_id = []
    # |1|-->Furthere Feature : Select Subject
    stages = ['2D_N1','2D_N2','2D_N3','2D_Rem','2D_Wake']
    file_list = []
    dir_re = []
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    
    mypath = './decompo_data'
    for (dirpath, dirnames, filenames) in walk(mypath):
        break
    
    subject_id = dirnames # Focus here For Featrue |1|
    
    for i in range(len(subject_id)):
        if(i==(test_subnum-1)):
            continue
        
        step_mypath = mypath + '/' +str(subject_id[i])
        for j in range(len(stages)) :
            sstep_mypath = step_mypath + '/' +str(stages[j])
            for (dirpath, dirnames, filenames) in walk(sstep_mypath):
                for k in range(len(filenames)):
                    dir_re.append(str(sstep_mypath) + '/'+ str(filenames[k]))
                    
    for i in range(len(dir_re)):
        processing_npz_name = dir_re[i]
        working_npz  = np.load(processing_npz_name)
        train_X.append(working_npz)
        ylabel = processing_npz_name[27:-10]
        if ylabel == 'N1':
            train_Y.append(0)
        elif ylabel == 'N2':
            train_Y.append(1)
        elif ylabel == 'N3':
            train_Y.append(2)
        elif ylabel == 'Rem':
            train_Y.append(3)
        elif ylabel == 'Wake':
            train_Y.append(4)


    ##############start of test data#############

    subject_id = []
    # Used_Subject_id = []
    # Used_Subject_id = []
    # |1|-->Furthere Feature : Select Subject
    stages = ['2D_N1','2D_N2','2D_N3','2D_Rem','2D_Wake']
    file_list = []
    dir_re = []
    
    mypath = './decompo_data'
    for (_dirpath, _dirnames, _filenames) in walk(mypath):
        break
    
    _subject_id = _dirnames # Focus here For Featrue |1|
    
    #i = int(len(subject_id1))-1   #Set Usage of Train Temp Develop Version!!
    for i in range(len(_subject_id)):
        if i == (test_subnum-1):
            print("##########################################################Test Subject Name:"+str(_subject_id[test_subnum-1])+'##########################################################')
            #print(i)
            _step_mypath = mypath + '/' +str(_subject_id[i])
            for j in range(len(stages)) :
                _sstep_mypath = _step_mypath + '/' +str(stages[j])
                for (_dirpath, _dirnames, _filenames) in walk(_sstep_mypath):
                    for k in range(len(_filenames)):
                        dir_re.append(str(_sstep_mypath) + '/'+ str(_filenames[k]))
        else :  pass
                    
    for i in range(len(dir_re)):
        processing_npz_name = dir_re[i]
        working_npz  = np.load(processing_npz_name)
        test_X.append(working_npz)
        ylabel = processing_npz_name[27:-10]
        if ylabel == 'N1':
            test_Y.append(0)
        elif ylabel == 'N2':
            test_Y.append(1)
        elif ylabel == 'N3':
            test_Y.append(2)
        elif ylabel == 'Rem':
            test_Y.append(3)
        elif ylabel == 'Wake':
            test_Y.append(4)
    if onehot_ == 1:        
        return train_X,train_Y,test_X,test_Y
    else :
        return train_X,dense_to_one_hot(train_Y),test_X,dense_to_one_hot(test_Y)


def Data_spliter():
    X,Y = Data()
