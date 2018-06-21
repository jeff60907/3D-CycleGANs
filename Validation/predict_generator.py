import os
import time
import numpy as np
import argparse
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.models import load_model 
from keras_contrib.layers.normalization import InstanceNormalization
import read_tools
import gc

resolution = 64
batch_size = 5

###############################################################
config={}

config['train_names'] = ['chair']
for name in config['train_names']:
    config['X_train_'+name] = '../Data/'+name+'/train_25d/'
    config['Y_train_'+name] = '../Data/'+name+'/train_3d/'


config['test_names']=['chair']
for name in config['test_names']:
    config['X_test_'+name] = '../Data/'+name+'/test_25d/'
    config['Y_test_'+name] = '../Data/'+name+'/test_3d/'

config['resolution'] = resolution
config['batch_size'] = batch_size

################################################################

# make chair40 training data dir  8x5  
def make_traing_data_dir(data):
    for _ in range(8):
        x_train_batch, Y_train_batch = data.load_X_Y_voxel_grids_train_next_batch()
        for name in data.batch_name:
            if not os.path.exists(name[24:-18]+'/'+name[24:-13]):
	        os.makedirs(name[24:-18]+'/'+name[24:-13])

# load data config
data = read_tools.Data(config)

make_traing_data_dir(data)


# load network to generate sparse2dense 3D models

data = read_tools.Data(config)
autoencoder = load_model('./saved_model/g_AB_chair.h5')
for _ in range(len(data.X_train_files)/batch_size):
    X_train_batch, Y_train_batch = data.load_X_Y_voxel_grids_train_next_batch()
    g = autoencoder.predict(X_train_batch)
    for j in range(batch_size):
	name = data.batch_name[j][24:-13]
	dir_name = name[:-5] + '/'+name
	print name[:-5] + '/'+name + '/fake_'+ name 
	data.output_Voxels(dir_name + '/fake_' + name, g[j])
	data.plotFromVoxels(dir_name + '/fake_' + name, g[j])
	#data.plotFromVoxels(dir_name + '/'+name + '/sparse_' + name, X_train_batch[j])
	#data.plotFromVoxels(dir_name + '/'+name + '/truth_' + name,Y_train_batch[j])

