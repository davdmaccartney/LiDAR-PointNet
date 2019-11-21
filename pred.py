import os
import argparse
import math
import numpy as np
import tensorflow as tf
import socket
from random import shuffle
from laspy.file import File
from mortonpy.morton import Morton
import json
import pandas as pd
from scipy.stats import binned_statistic_dd
import sys
import tf_util
from model import *
import logging


pd.options.display.float_format = '{:.6f}'.format
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
FLAGS = parser.parse_args()

BATCH_SIZE = 1
POINT_DIM = 8
NUM_CLASSES = 3
GPU_INDEX = FLAGS.gpu
NUM_POINT = 4096

Chemin='C:/LiDAR/test'

m = Morton(dimensions=2, bits=32)

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
file_name =  os.path.basename(sys.argv[0])

os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp '+file_name+ ' %s' % (LOG_DIR)) # bkp of model def # bkp of train procedure

    
class style():
    BLACK = lambda x: '\033[30m' + str(x)
    RED = lambda x: '\033[31m' + str(x)
    GREEN = lambda x: '\033[32m' + str(x)
    YELLOW = lambda x: '\033[33m' + str(x)
    BLUE = lambda x: '\033[34m' + str(x)
    MAGENTA = lambda x: '\033[35m' + str(x)
    CYAN = lambda x: '\033[36m' + str(x)
    WHITE = lambda x: '\033[37m' + str(x)
    UNDERLINE = lambda x: '\033[4m' + str(x)
    RESET = lambda x: '\033[0m' + str(x)

def get_hash(row):
    return m.pack(row['mX'],row['mY'])

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()


def get_a_model():
    #with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT, POINT_DIM)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())

            pred = get_model(pointclouds_pl, is_training_pl, POINT_DIM)
            #pred_softmax = tf.nn.softmax(pred)
            loss = get_loss(pred, labels_pl)
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, tf.train.latest_checkpoint('D:/Pointnet-python/Train/log/', latest_filename=None))
        #saver.restore(sess, CHECKPOINT)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred}

        return sess, ops

def inference(sess, ops, data):
    feed_dict = {ops['pointclouds_pl']: data,
                 ops['is_training_pl']: False,}
    pred_val = sess.run(ops['pred'],feed_dict=feed_dict)

    pred_val = np.argmax(pred_val, 2)
    return pred_val

def load_data_memory(file):
    df = pd.DataFrame(np.empty(0, dtype=[('Xx',np.float),('Yy',np.float),('Zz',np.float),('intensity',np.int), ('raw_classification',np.int), ('gps_time',np.int32), ('elevation',np.float),('return',np.int),('X',np.int), ('Y',np.int), ('Z',np.int), ('hash',np.int), ('XN',np.float), ('YN',np.float), ('ZN',np.float) ]))
    las_data = File(full_path, mode='r')
    df['Xx'] = (las_data.x)
    df['Yy'] = (las_data.y)
    df['Zz'] = (las_data.z)
    df['intensity'] = (las_data.intensity)
    df['raw_classification'] = (las_data.raw_classification)
    df['gps_time'] = (las_data.gps_time)
    df['elevation'] = (las_data.user_data) / 10
    df['returns'] = (las_data.return_num)
    
    # Density
    Density = len(df.index) / ((df.Xx.max() - df.Xx.min()) * (df.Yy.max() - df.Yy.min()))
    print(style.GREEN('Density :%f pts/m2'% Density))
    grid_size = math.sqrt(NUM_POINT / Density)
    print('grid_size :%i m' % grid_size)

    # origine translation
    print('translation')
 
    df['X'] = df.Xx - df.Xx.min()
    df['Y'] = df.Yy - df.Yy.min()
    df['Z'] = df.Zz - df.Zz.min()

    # hash
    print('hash')
    df['mX'] = df.X // grid_size
    df.mX = df.mX.astype(int)
    df['mY'] = df.Y // grid_size
    df.mY = df.mY.astype(int)
 
    df['hash'] = df[['mX', 'mY']].apply(get_hash, axis=1)
    del df['mX']
    del df['mY']

    # normallisation
    print('normallisation')
    df['XN'] = (df.Xx - df.Xx.min()) / (df.Xx.max() - df.Xx.min())
    df['YN'] = (df.Yy - df.Yy.min()) / (df.Yy.max() - df.Yy.min())
    df['ZN'] = (df.Zz - df.Zz.min()) / (df.Zz.max() - df.Zz.min())

    df['intensity'] = (df.intensity - df.intensity.min()) / (df.intensity.max() - df.intensity.min())
    df['elevation'] = (df.elevation - df.elevation.min()) / (df.elevation.max() - df.elevation.min())
    df['returns'] = (df.returns - df.returns.min()) / (df.returns.max() - df.returns.min())
    df = df.sort_values(by=['hash'])
    
    return df

if __name__=='__main__':
 with tf.Graph().as_default():
  
   cls()  
   # Boucle sur les Laz
   for filename in os.listdir(Chemin):
    if filename.endswith(".laz"):
     full_path = os.path.join(Chemin, filename)
     nom = os.path.splitext(filename)[0]
    
     LOG_FOUT = open(os.path.join(LOG_DIR, nom+'.log'), 'w')
     LOG_FOUT.write(str(FLAGS) + '\n')

     print(style.CYAN('runing : ' + nom))
     df = load_data_memory(filename)
  

     batches = [df[i:i + NUM_POINT] for i in range(0,len(df),NUM_POINT)]
     if len(batches[-1]) < NUM_POINT: batches = batches[:-1]
     print(style.MAGENTA('Batches : %i'% len(batches)))


     data_channels = ['X', 'Y', 'Z','intensity','elevation', 'XN','YN','ZN']

     data_XYZ = ['Xx', 'Yy','Z']
     data_GPS_Time = ['gps_time']
     data_intensity = ['intensity']
  
     num_batches = 0
     sess, ops = get_a_model()
     cls() 
     f = open(os.path.join(LOG_DIR, nom+'.txt'), 'w')
     for lot in batches:
        
         np.set_printoptions(suppress=True)
         data = np.empty([0, NUM_POINT, POINT_DIM], dtype=float)
         dataXYZ = np.empty([0, NUM_POINT, 3], dtype=float)
         dataGPS = np.empty([0, NUM_POINT, 1], dtype=float)
         dataINT = np.empty([0, NUM_POINT, 1], dtype=float)
      
         s = [lot[data_channels].values]
         ss = np.round_(s, decimals=6)

         s2 = [lot[data_XYZ].values]
         sXYZ = np.round_(s2, decimals=6)            
      
         sGPS = [lot[data_GPS_Time].values]
         sInt = [lot[data_intensity].values]
      
         dataINT = np.concatenate([dataINT, sInt]) 
         dataGPS = np.concatenate([dataGPS, sGPS])  
         dataXYZ = np.concatenate([dataXYZ, sXYZ])
         data = np.concatenate([data, ss])

         dataINT = np.where(np.isnan(dataINT), 0, dataINT)
         dataGPS = np.where(np.isnan(dataGPS), 0, dataGPS)

         num_batches += 1
         log_string('**** num_batches %03d ****' % (num_batches))

         if num_batches % 100 == 0:
          print(style.GREEN('Current batch num : %i on %i' % (num_batches,len(batches))))
        
         pred_val = inference(sess, ops, data)

         for j in range(NUM_POINT):
             f.write('%f %f %f %f %.8f %i\n' % (dataXYZ[0,j,0], dataXYZ[0,j,1], dataXYZ[0,j,2], dataINT[0,j], dataGPS[0,j], pred_val[0,j]))
          
         f.flush()  

    
     df.drop(df.index, axis=0, inplace=True)
     del batches
     LOG_FOUT.close()
     f.close()
     nomlas = os.path.join(LOG_DIR, nom+'.las')
     nomtxt = os.path.join(LOG_DIR, nom+'.txt')
     txt2las = 'C:/LAStools/bin/txt2las -i ' + nomtxt + ' -o ' + nomlas + ' -olaz -parse xyzitc -cpu64'
     os.system(txt2las)
     print('\n') 
  
   del df
   print('Done!')
