import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#mel_process
def mlp_preprocess(dev_arr, dev_lab_arr, k, step_size = 1):

  temp_flat_dev = list()
  temp_flat_dev_lab = list()

  for utter_idx in range(dev_arr.shape[0]):
    temp_flat_dev_lab += dev_lab_arr[utter_idx].tolist()
    for frm in dev_arr[utter_idx]:
      temp_flat_dev.append(frm.tolist())

  print("Number of Frames: {}".format(len(temp_flat_dev)))
  print("Number of Frame Labels: {}".format(len(temp_flat_dev_lab)))

  temp_flat_dev = [[0]*40]*k + temp_flat_dev + [[0]*40]*k

  re_temp_flat_dev = list()
  re_temp_flat_dev_lab = list()

  for frm_idx in range(k, len(temp_flat_dev)-k, step_size): # Default: k=0, step_size=1
    temp_k_gram_vec = list()
    ## temp_k_gram_vec_lab = 0
    for i in range(-k, k+1, 1):
      temp_k_gram_vec += temp_flat_dev[frm_idx+i]


    re_temp_flat_dev.append(temp_k_gram_vec)
 

  print("Number of k-gram Frames: {}".format(len(re_temp_flat_dev)))
  print("Number of k-gram Frame Labels: {}".format(len(temp_flat_dev_lab))) ##

  new_dev_arr = np.array(re_temp_flat_dev)
  print("New dev_arr shape: {}".format(new_dev_arr.shape))
  new_dev_arr_lab = np.array(temp_flat_dev_lab)
  print("New dev_arr_lab shape: {}".format(new_dev_arr_lab.shape)) ##

  return new_dev_arr, new_dev_arr_lab

#model_pred
def model_pred(X, y, test_data, epochs_ = 50, batch_size_=250, verbose_=1, test_size_ = 0.33):
  # ensure all data are floating point values
  X = X.astype('float32')
  # split into train and test datasets
  np.random.seed(1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
  # determine the number of input features
  n_features = X_train.shape[1]
  # define model
  model = Sequential()
  model.add(Dense(120*3, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
  model.add(Dense(138*3, activation='relu', kernel_initializer='he_normal'))
  model.add(Dense(138*2, activation='relu', kernel_initializer='he_normal'))
  model.add(Dense(138, activation='softmax'))
  # compile the model
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  # fit the model
  history = model.fit(X_train, y_train, epochs=epochs_, batch_size=batch_size_, verbose=verbose_, validation_data = (X_test, y_test))
  pred = model.predict(X_test)
  pred = np.argmax(pred, axis = 1)
  #etst data predction
  test_pred = model.predict(test_data)
  test_pred = np.argmax(test_pred, axis = 1)
  
  return model, history , pred , test_pred

  #


#run
def main(hyper):
    dev = hyper['dev_data']
    dev_lables = hyper['dev_labels']

    dev = np.load(dev,allow_pickle=True, encoding="bytes")
    dev_labels = np.load(dev_labels,allow_pickle=True, encoding="bytes")
    
    k = hyper['k_val_int']
    epochs_ = hyper['epochs']
    test_size = hyper['test_size']
    test_data = hyper['test_data']
    submission = hyper['submission']

    X_1 , y_1 = mlp_preprocess(dev , dev_labels , k)

    model , history , predction, test_pred = model_pred(X_1 , y_1 , test_data = test_data epochs_= epochs_ , test_size_= test_size)


    pred_arr = test_pred
    df = pd.DataFrame(pred_arr)
    df.to_csv(submission, index = False)
    

    return df 

if __name__ = '__main__':
    hyper = {
        'epochs' : 50
        'test_size' : 0.35
        'k_val_int' : 5
        'dev_data' : './drive'
        'dev_labels' : './drive'
        'test_data' : './drive'
        'submission' : ',/drive'
    }
    main(hyper)
    
    


