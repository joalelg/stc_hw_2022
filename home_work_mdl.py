#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets

import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics


from scipy.stats import entropy
import math

import torch
from torchvision import  models
import torchvision

#%%
# ======= Task 1. ======= 

ROOT_PATH = os.path.dirname(os.getcwd())
RANDOM_SEED = 123


def drop_labels(dataset_labels, proportion):
    """Drop a desired proportion of labels from a labels list   .

    Args:
        dataset_labels (list): list of integers
        proportion (float): proportion of tags to be dropped

    Returns:
        pandas data frame: Original labels and in sample boolean tag
        ['dataset_labels', 'in_sample']
    """
    
    df = pd.DataFrame(dataset_labels, columns=['dataset_labels'])
    df['in_sample'] = True
    indexes_to_remove = df \
                            .sample(frac=proportion,
                                    axis=0,
                                    random_state = RANDOM_SEED 
                                    ).index

    df.loc[indexes_to_remove, 'in_sample']=False

    
    return df

    

# %%
# ======= TASK 2 - Data cleaning ======= 

from PIL import Image

images_df = pd.DataFrame(None, columns=['data_set', 'filename', 'mode', 'size'])

train_dir = os.path.join(ROOT_PATH, 'data','cars_train')
test_dir = os.path.join(ROOT_PATH, 'data','cars_test')

idx = 0
for directory in [train_dir, test_dir]: 
    for file in os.listdir(directory):
        
        filename = os.fsdecode(file)
        image_path = os.path.join(directory, filename)
        with Image.open(image_path) as image_pil:
            mode, size = image_pil.mode, image_pil.size
        dir_name = os.path.basename(directory)
        
        # Populate df with image filename, mode and size   
        images_df.loc[idx,:] = dir_name, filename, mode, str(size)
        
        if mode != 'RGB':  # Remove unwanted imges
            print(f'Removing image {idx} {dir_name}/{filename}')
            os.remove(image_path)
   
        idx += 1            
            
        


# %%
# Some images are in gray, they have image_pil.mode 'L' instead of 'RGB'
print(images_df.groupby('data_set')['mode'].value_counts()) 
print(images_df.head())

# data_set    mode
# cars_test   RGB     8025
#             L         16
# cars_train  RGB     8126
#             L         18

        
# %%

#%%
# ======= TASK 3 - Dataset representation ======= 

# %%
# Load train set labels and merge them with images_df
labels = pd.read_csv(os.path.join(ROOT_PATH, 'data', 'train_perfect_preds.txt')
                     ,header=None, names=['labels'])
labels['data_set'] = 'cars_train'
labels['labels'] = labels['labels'].astype(str)

labels = labels.reset_index().rename({'index':'filename'}, axis=1)
labels['filename'] = labels['filename'].apply(lambda x:  f'{x+1:05d}.jpg')


# Merge with images df
images_tags_df = images_df.merge(labels, on=['data_set','filename'], how='left')

# Set up the dictionary with required structure
# {1: {'embedding': <np.ndarray>, 'class_idx': <int>, ‘labelled': <boolean or int>
embeddings_dic = {}

# Load Resnet18 model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
# print(list(model.children()))

#%%
# Populate the dictionary

embeddings_dic = {}

idx = 1
total_files = len(os.listdir(directory))

# for directory in [train_dir]: 
#     for file in os.listdir(directory):        
                    
#         filename = os.fsdecode(file)
#         image_path = os.path.join(directory, filename)
#         with Image.open(image_path) as image_pil:
#             mode, size = image_pil.mode, image_pil.size
#             image_tensor = torchvision.transforms.functional.to_tensor(image_pil)
#             embedding_torch = model(image_tensor.unsqueeze(0))
#             embedding_np = embedding_torch.detach().cpu().numpy()
            
#         dset_mask = images_tags_df['data_set']=='cars_train'
#         filename_mask = images_tags_df['filename']==filename
#         class_label = images_tags_df.loc[dset_mask & filename_mask,'labels'].values[0]        
#         embeddings_dic[idx]={'embedding':embedding_np,
#                              'class_idx':class_label,
#                              'labelled':1} 
        
#         if (idx-1)%500==0: # Output partial completion
#             print(f'embedding image {idx} of {total_files}....')               
#         idx+=1

# # Save dicionary using torch
# dic_path = os.path.join(ROOT_PATH, 'data',  'embeddings.pt')
# torch.save(embeddings_dic, dic_path)
#%%
# ======= TASK 4 - Build a partially labelled dataset ======
dic_path = os.path.join(ROOT_PATH, 'data',  'embeddings.pt')
full_data = torch.load(dic_path)
print(f'Loaded data size: {len(full_data.keys())}')


all_labels = [int(full_data[k]['class_idx']) for k in full_data.keys()]
labels_dropped = drop_labels(all_labels, proportion=0.6)
#labels_dropped['in_sample'] = labels_dropped['in_sample']*1 
print(labels_dropped.shape)
print(labels_dropped.head())
#%%
# Create data set with 40% of the data being labeled
sample_data = {}
for k,include in zip(full_data.keys(), labels_dropped['in_sample']):
    if include:
        sample_data[k] = {kk:v for kk,v in full_data[k].items()}
    else:
        continue

print(f'Sample data size: {len(sample_data.keys())}') 
sample_porportion = len(sample_data.keys()) / len(full_data.keys())
print(f'New data labelled proportion: {sample_porportion:.4}')
#%%
# ======= TASK 5 - Create train/validation split ======= 

def split_data(datase_inputs, dataset_labels, training_proportion = 0.8):
    """Function to split data into train and test subsets.

    Args:
        datase_inputs iterable: Input explanatory features
        dataset_labels iterable: Category tags
        training_proportion (float, optional): Train set proportion size. Defaults to 0.8.

    Returns:
        iterables: X_train, X_test, y_train, y_test data subsets corresponding to 
         training_inputs, test_inputs, training_labels, test_labels.
    """
    X, y = datase_inputs, dataset_labels
    X_train, X_test, y_train, y_test = train_test_split( X, y,
                                            test_size=1-training_proportion,
                                            random_state=RANDOM_SEED
                                        )

    X_train, X_test = np.stack(X_train), np.stack(X_test) # Convert to arrays     
    # Slice to reshape from                                   
    return X_train[:, 0, :], X_test[:, 0, :], np.array(y_train), np.array(y_test)


inputs = [sample_data[k]['embedding']for k in sample_data.keys()]
labels = [sample_data[k]['class_idx'] for k in sample_data.keys()]
# print(len(inputs), len(labels))

X_train, X_test, y_train, y_test = split_data(inputs, labels, training_proportion = 0.8)    

#%%
# ======= TASK 6Experiment(s) to convince clients that more labels will improve model performance = 


X = X_train
y = y_train


clf = make_pipeline(#StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3,
                    loss = 'log'))

clf.fit(X, y)
#%%

# Make predictions and get performance metrics

train_preds = clf.predict(X) #[[-0.8, -1]]
train_probas = clf.predict_proba(X)
test_preds  = clf.predict(X_test)
test_pobas  = clf.predict_proba(X_test)

metrics.precision_score(y_test, test_preds, average='micro')

#%%
# ======= TASK 7 - Active learning to select new instances to be labelled =======


# Calculate entropy of the full data set
all_embeddings_np = np.array([full_data[k]['embedding'] for k in full_data.keys()])[:, 0, :]


# Predict probabiities on full data set
full_probas = clf.predict_proba(all_embeddings_np)
full_probas.shape

# Calculate entropies 
entropies = np.array([entropy(p) for p in full_probas])
entropies.shape

id_position_pd = pd.DataFrame({'id':list(full_data.keys()),
                               'position': np.argsort(entropies)[::-1],
                               'entropy': entropies})


# Calculate the number of K of examples to add
actual_size = len(sample_data)
target_size = math.ceil(len(full_data)*.65)
n_top = target_size - actual_size
print(f'Actual and target sizes: {actual_size} {target_size}. To add: {n_top}')



#%%
# Get keys not being in the sample
in_sample_ids = [i for i in sample_data.keys()]
out_sample_ids = set(id_position_pd['id']).difference(set(in_sample_ids))

len(set(in_sample_ids).intersection(out_sample_ids)) # Check 0 intersect

top_out_sample_pd = id_position_pd \
                        .loc[id_position_pd['id'].isin(out_sample_ids),:] \
                        .sort_values(by='position') \
                        .head(n_top)                            


top_out_sample_ids = sorted(top_out_sample_pd['id'])          
top_out_sample_ids[0:4]

extended_sample_ids = sorted([i for i in set(in_sample_ids).union(set(top_out_sample_ids))])


#%%
# Then create the new data set

sample_data2 = {}
for i in extended_sample_ids:
        sample_data2[i] = {k:v for k,v in full_data[i].items()}

# list(sample_data2.keys())[0:15], len(sample_data2)
#%%
# ======= TASK 8 - Final Model Evaluation =======

inputs = [sample_data2[k]['embedding']for k in sample_data2.keys()]
labels = [sample_data2[k]['class_idx'] for k in sample_data2.keys()]
# print(len(inputs), len(labels))

X_train, X_test, y_train, y_test = split_data(inputs, labels, training_proportion = 0.8)    



X = X_train
y = y_train


clf = make_pipeline(#StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3,
                    loss = 'log'))

clf.fit(X, y)
# %%
# Make predictions and get performance metrics

train_preds = clf.predict(X) #[[-0.8, -1]]
train_probas = clf.predict_proba(X)
test_preds  = clf.predict(X_test)
test_pobas  = clf.predict_proba(X_test)

metrics.precision_score(y_test, test_preds, average='micro')


# %%
