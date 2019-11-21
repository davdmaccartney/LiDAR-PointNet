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
import fnmatch
import os
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

parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')

parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--n_augmentations', type=int, default=1, help='Number of augmentations option: 1-6 [default: 1]')

parser.add_argument('--pred', type=float, default=0.85, help='last pred [default: 0.60]')


FLAGS = parser.parse_args()

eval_global_pred = FLAGS.pred
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
N_AUGMENTATIONS = FLAGS.n_augmentations

file_name =  os.path.basename(sys.argv[0])
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('copy model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp '+file_name+ ' %s' % (LOG_DIR)) # bkp of model def # bkp of train procedure
os.system('color')

POINT_DIM = 8
MAX_NUM_POINT = 4096
NUM_CLASSES = 3
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5

BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

current_best_eval_pred=np.float
best_eval_loss_name='default'
numBatches = 0
mesageRun = []
mesageEpoch = []
mesageTrain = []
mesageEval = []
m = Morton(dimensions=2, bits=32)

def generator(df, hashes, BATCH_SIZE, NUM_POINT, N_AUGMENTATIONS, shuffled=True):

    data_channels = ['X', 'Y', 'Z','intensity', 'elevation','XN','YN','ZN']

    seed_hash = []


    if len(hashes)<BATCH_SIZE: N_AUGMENTATIONS = math.ceil(BATCH_SIZE / len(hashes))

    for seed in range(N_AUGMENTATIONS):
        #print(seed)
        for h in hashes:
            seed_hash.append((seed, h))
    shuffle(seed_hash)
    
    
    

    batches = [seed_hash[i:i+BATCH_SIZE] for i in range(0,len(seed_hash),BATCH_SIZE)]
    if len(batches[-1]) < BATCH_SIZE: batches = batches[:-1]
    if shuffled: [shuffle(batch) for batch in batches]   
    

    def random_sample_block(group, seed):
        """
        Sample entirely random for the entire grid cell
        IN: group (all points in a grid cell), seed (random state value)
        OUT: data_group (a subset of the points in the grid cell; a training sample)
        """    
        if len(group) > NUM_POINT:
            data_group = group.sample(n=NUM_POINT, replace=False, random_state=seed)
        else:
            data_group = group.sample(n=NUM_POINT, replace=True, random_state=seed)
        
        return data_group     
        
    global numBatches
    numBatches = len(batches)
    
    for batch in batches:
        df_batch =[random_sample_block(df[df['hash']==h],s) for s,h in batch]
        data = np.empty([0, NUM_POINT, POINT_DIM], dtype=float)
        label = np.empty([0, NUM_POINT], dtype=int)
        if len(df_batch)==BATCH_SIZE:
                for b in df_batch:
                         s = [b[data_channels].values]
                         ss = np.round_(s, decimals=6)
                         if ss.shape[1] == NUM_POINT:
                            data = np.concatenate([data, ss])
                            label = np.concatenate([label, [b.raw_classification.values]])
                         else:  sys.exit(0)
                yield data, label
        else: sys.exit(0)
    
           
        
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()

def get_learning_rate(batch):
    learning_rate = tf.compat.v1.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def get_hash(row):
    return m.pack(row['mX'],row['mY'])

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

    
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

def update_progress(progress):
    os.system('cls' if os.name=='nt' else 'clear')
    barLength = 30 # Modify this to change the length of the progress bar
    global mesageRun, mesageEpoch, mesageTrain, mesageEval    
    block = int(round(barLength*progress))
    text = "\rRun-Progresse: [{0}] {1}% ".format( "#"*block + "-"*(barLength-block), int(progress*100))
    sys.stdout.write(text)
    print(style.YELLOW('\n'))
    sys.stdout.write('\n'.join(map(str, mesageRun)))
    print(style.MAGENTA('\n'))
    sys.stdout.write('\n'.join(map(str, mesageEpoch)))
    print(style.CYAN('\n'))
    sys.stdout.write('\n'.join(map(str, mesageTrain)))
    print(style.GREEN('\n'))
    sys.stdout.write('\n'.join(map(str, mesageEval)))
    sys.stdout.flush()


def get_a_model():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT, POINT_DIM)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)

            print('get model')
            pred = get_model(pointclouds_pl, is_training_pl, POINT_DIM, bn_decay)
            loss = get_loss(pred, labels_pl)
            tf.compat.v1.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.cast(labels_pl, tf.int64))
            #correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.compat.v1.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            saver = tf.compat.v1.train.Saver()    

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

       # saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))
        print ("Model Setup")  

        ops = {
                   'pointclouds_pl': pointclouds_pl,
                   'labels_pl': labels_pl,
                   'is_training_pl': is_training_pl,
                   'pred': pred,
                   'loss': loss,
                   'train_op': train_op,
                   'merged': merged,
                   'step': batch}

        return sess, ops, train_writer, test_writer, train_writer, test_writer, saver  



def load_data_memory(full_path):
      
    nom = os.path.basename(full_path)
    print(style.CYAN(nom))
    df = pd.DataFrame(np.empty(0, dtype=[('Xx',np.float),('Yy',np.float),('Zz',np.float),('intensity',np.int), ('raw_classification',np.int), ('gps_time',np.int32), ('elevation',np.float),('return',np.int),('X',np.int), ('Y',np.int), ('Z',np.int), ('hash',np.int), ('XN',np.float), ('YN',np.float), ('ZN',np.float) ]))
    las_data = File(full_path, mode='r')

    df['Xx'] = (las_data.x)
    df['Yy'] = (las_data.y)
    df['Zz'] = (las_data.z)
    df['intensity'] =(las_data.intensity)
    df['raw_classification'] =(las_data.raw_classification)
    df['gps_time'] =(las_data.gps_time)
    df['elevation'] =(las_data.user_data)/10
    df['returns'] =(las_data.return_num)
  
  
    # Density
    Density =  len(df.index) / ((df.Xx.max() - df.Xx.min()) * (df.Yy.max() - df.Yy.min()))
    print(style.GREEN('Density :%f pts/m2'% Density))
    grid_size = math.sqrt(NUM_POINT/Density)
    print('grid_size :%i m'% grid_size)
    mesageRun.append('Density :%f pts/m2'% Density)
    mesageRun.append('Grid_size :%i m' % grid_size)
    mesageRun.append('Preparing Data')
    
    # origine translation
    print('translation')
    log_string('Density :%f'% Density)
    log_string('Grid_size :%i' % grid_size)

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


    # memory optimisation
    print('train / validation / test')
    hash_centers = df.sort_values(by=['hash'])
    # train/test
    hashes = hash_centers['hash'].unique()
    shuffle(hashes)
    #train_test_msk = np.random.rand(len(hashes))
    L = int(len(hashes)*.8)
    train_hashes, validation_hashes = hashes[:L], hashes[L:]
    
    with open(os.path.join(LOG_DIR,filename+'_data_split.json'), 'w') as data_split:
        json.dump({'train': train_hashes.tolist(), 'validation': validation_hashes.tolist()}, data_split)
    print('loading done')
    #print('\n')
    del hashes

    return df, train_hashes, validation_hashes

def train(df, sess, ops, train_hashes, validation_hashes, train_writer, test_writer, saver, nom):
    mesageRun.clear()
    # boucle de travail
    for epoch in range(MAX_EPOCH):
          global eval_global_pred
          global best_eval_loss_name
          global mesageEpoch
          mesageRun.clear()
          mesageEpoch.clear()
          mesageRun.append('Best predict :%f'% eval_global_pred+' --> '+ best_eval_loss_name)
          mesageRun.append('Run : '+nom+' n°'+str(i)+' on '+str(total_con))
          mesageEpoch.append('EPOCH ' +str(epoch)+ ' on '+str(MAX_EPOCH))
          log_string('EPOCH %03d ' % (epoch))
 
          train_one_epoch(df, sess, ops, train_writer, train_hashes)

          if epoch % 5 == 0:
                 eval_one_epoch(df, sess, ops, test_writer, validation_hashes)
          else: 
                 mesageEval.clear()

          
          if current_best_eval_pred > eval_global_pred:
                 best_eval_loss_name = nom
                 eval_global_pred = current_best_eval_pred
                 mesageRun.append('Best predict :%f'% eval_global_pred+' --> '+ best_eval_loss_name)
                 save_path = saver.save(sess, os.path.join(LOG_DIR, best_eval_loss_name + "_best_model_epoch_%03d.ckpt"  % (epoch)),)
                 log_string("Model saved in file: %s" % save_path)

          if epoch % 10 == 0:
                 save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step= ops['step'])
                 log_string("Model saved in file: %s" % save_path)

          


def train_one_epoch(df, sess, ops, train_writer, train_hashes):
    is_training = True
    global mesageTrain
    mesageTrain.clear()
    mesageTrain.append('----')
    mesageTrain.append('train')
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    num_batches = 0


    global numBatches
    for batch_data, batch_label in generator(df, train_hashes, BATCH_SIZE, NUM_POINT, N_AUGMENTATIONS):

        num_batches += 1 * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        update_progress(num_batches/(numBatches*BATCH_SIZE))

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
    mesageTrain.append('mean loss: %f' % (loss_sum / float(num_batches)))
    mesageTrain.append('accuracy: %f' % (total_correct / float(total_seen)))


    update_progress(num_batches/(numBatches*BATCH_SIZE))

        
def eval_one_epoch(df, sess, ops, test_writer, validation_hashes):
    """ ops: dict mapping from string to tf ops """
    global numBatches
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    global mesageEval
    mesageEval.clear()
    mesageEval.append('----')
    mesageEval.append('eval')
    
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('----')

    num_batches = 0

    for batch_data, batch_label in generator(df, validation_hashes, BATCH_SIZE, NUM_POINT, N_AUGMENTATIONS):
        num_batches += 1 * BATCH_SIZE
       

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        update_progress(num_batches/(numBatches*BATCH_SIZE))
        for i in range(BATCH_SIZE):
            for j in range(NUM_POINT):
                l = batch_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i, j] == l)

    
    global current_best_eval_pred
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    
    if total_correct == 0: 
        Tcorect = total_correct / 0.0001
        log_string('eval accuracy : %f'% Tcorect)
    else:
        Tcorect = total_correct / float(total_seen)
        log_string('eval accuracy : %f'% Tcorect)
    
    mesageEval.append('mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    mesageEval.append('accuracy : %f'% Tcorect)

    CC = 'Predic classes :'+str(total_correct_class)
    SC = 'Actual classes :'+str(total_seen_class)
    mesageEval.append(CC)
    mesageEval.append(SC)
    update_progress(num_batches/(numBatches*BATCH_SIZE))

    log_string(CC)
    log_string(SC)
        
    for i in range(NUM_CLASSES):
       if total_correct_class[i] == 0:total_correct_class[i] = 1

    avg = np.mean(np.array(total_correct_class)/np.array(total_seen_class),dtype=np.float)
    current_best_eval_pred = avg
    log_string('eval avg class acc: %f' % (avg))
    mesageEval.append('avg class acc: %f' % (avg))
    update_progress(num_batches/(numBatches*BATCH_SIZE))



if __name__ == "__main__":
 with tf.Graph().as_default():
  cls()

  Chemin='C:/LiDAR/'

  i = 0
  total_con=len(fnmatch.filter(os.listdir(Chemin), '*.laz'))
  sess, ops, train_hashes, validation_hashes, train_writer, test_writer, saver = get_a_model()
  cls()
  # Boucle sur les Laz
  for filename in os.listdir(Chemin):
   if filename.endswith(".laz"):
    i = i+1
    full_path = os.path.join(Chemin, filename)
    nom = os.path.splitext(filename)[0]
    LOG_FOUT = open(os.path.join(LOG_DIR, nom + '.log'), 'w')
    LOG_FOUT.write(str(FLAGS)+'\n')
    mesageRun.clear()
    mesageRun.append('runing : '+nom+' : on '+str(i)+' / '+str(total_con))
    df, train_hashes, validation_hashes = load_data_memory(full_path)
    train(df, sess, ops, train_hashes, validation_hashes, train_writer, test_writer, saver, nom)    
    update_progress(i/total_con)
    LOG_FOUT.write('--------------------------------------------'+'\n')
    LOG_FOUT.write(str(('Best loss :%f'% eval_global_pred))+' --> '+ best_eval_loss_name+'\n')
   cls()
  LOG_FOUT.close()
  print('Best loss :%f'% eval_global_pred+' --> '+ best_eval_loss_name)
  print('Done!')
