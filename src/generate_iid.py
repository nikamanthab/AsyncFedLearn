import pandas as pd
import numpy as np
import os

def generate_train_data(number_of_samples):
    label_dict = dict([(j,i) for (i,j) in list(enumerate(os.listdir('../Dataset/CIFAR-10-images/train/')))])
    print(label_dict)

    paths = []
    label = []
    for i in os.listdir('../Dataset/CIFAR-10-images/train/'):
        for j in os.listdir('../Dataset/CIFAR-10-images/train/'+str(i))[:500]:
            paths.append('../Dataset/CIFAR-10-images/train/'+str(i)+'/'+str(j))
            label.append(i)
    train_df = pd.DataFrame({"paths": paths, "label": label})  
    train_df = train_df.sample(frac=1)
    total_length = len(train_df)
    lengths = [total_length*i//100 for i in number_of_samples]
    dfs = []
    for i in lengths:
        dfs.append(train_df[:i])
        train_df = train_df[i:]
    return dfs

def generate_test_data():
    paths = []
    label = []
    for i in os.listdir('../Dataset/CIFAR-10-images/test/'):
        for j in os.listdir('../Dataset/CIFAR-10-images/test/'+str(i))[:500]:
            paths.append('../Dataset/CIFAR-10-images/test/'+str(i)+'/'+str(j))
            label.append(i)
    test_df = pd.DataFrame({"paths": paths, "label": label})
    return test_df


