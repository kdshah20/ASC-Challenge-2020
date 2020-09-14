# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:10:04 2020

@author: shahk
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

#import VisualizeNN as VisNN
#from sknn.mlp import Regressor, Layer

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


tdata = np.loadtxt("C:/Users/shahk/Documents/ASC Sim 2019/Input data/laminate_data.csv",delimiter=',');

X = tdata[:,0:9]
y = tdata[:,10]

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)


def norm_std(x):
  return (x - np.mean(train_X,axis=0)) / np.std(train_X,axis=0)

def norm(x):
  return (x/np.max(x))

def reverse_target(pred, mean, std): 
    return np.asarray(pred*std + mean)

normed_train_data = norm_std(train_X)
normed_test_data = norm_std(test_X)

normed_target = (train_y - np.mean(train_y))/np.std(train_y)
normed_test_target = (test_y - np.mean(train_y))/np.std(train_y)

# Building a neural network structure

tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)

def build_model():
  model = keras.Sequential([
    layers.Dense(9, activation='relu', input_shape=[train_X.shape[1]]),
    layers.Dense(9, activation='relu'),
    #layers.Dense(9, activation='sigmoid'),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  
  return model
  
# Running the model
model = build_model()

model.summary()

example_batch = train_X[:10]
example_result = model.predict(example_batch)
example_result

# No. of iterations
EPOCHS = 1500

# Fitting the model
history = model.fit(
  normed_train_data, normed_target,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])

# Retrieving data from the fitted model 
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#Plotting the MAE error and MSE error
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': history}, metric = "mae")
#plt.ylim([0, 10])
plt.ylabel('MAE')

plotter.plot({'Basic': history}, metric = "mse")
#plt.ylim([0, 20])
plt.ylabel('MSE')

# Finding the model performance on the test data
loss, mae, mse = model.evaluate(normed_test_data, normed_test_target, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPa".format(mae))

# Plotting the true predictions vs. test predictions
test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(normed_test_target*np.std(train_y)+np.mean(train_y), test_predictions*np.std(train_y)+np.mean(train_y))
plt.xlabel('True Values [Strength]')
plt.ylabel('Predictions [Strength]')
lims = [0, 1300]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

# Plotting the error on the test data
error = test_predictions - normed_test_target
plt.hist(error, bins = 20)
plt.xlabel("Prediction Error [Strength]")
_ = plt.ylabel("Count")

os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

final_score = []
shuff_pred = []

for i in range(test_X.shape[1]):

    # shuffle column
    shuff_test = normed_test_data.copy()
    shuff_test[:,i] = np.random.permutation(shuff_test[:,i])
    
    # compute score
    score = mean_absolute_error(test_y, reverse_target(model.predict(shuff_test).ravel(), np.mean(train_y), np.std(train_y)))
    shuff_pred.append(reverse_target(model.predict(shuff_test).ravel(), np.mean(train_y), np.std(train_y)))
    final_score.append(score)
    
final_score = np.asarray(final_score)
final_score

real_pred = reverse_target(model.predict(normed_test_data).ravel(), np.mean(train_y), np.std(train_y)) 
MAE = mean_absolute_error(test_y, real_pred)

plt.bar(range(train_X.shape[1]), (final_score - MAE)/MAE*100)
plt.xticks(range(train_X.shape[1]), ['L1','L2','L3','L4','L5','L6','L7','L8','Eff_stiff'])
np.set_printoptions(False)



quasi_iso = np.array([45,90,45,0,0,45,90,45,61000])
quasi_iso = norm_std(quasi_iso)
k = model.predict(quasi_iso.reshape(1,9))
k[0][0]*np.std(train_y)+np.mean(train_y)

uni = np.array([0,0,0,0,0,0,0,0,170000])

test_uni = norm_std(uni)
l = model.predict(uni.reshape(1,9))
l[0][0]*np.std(train_y)+np.mean(train_y)


network_structure = np.hstack(([train_X.shape[1]], np.asarray(clf.hidden_layer_sizes), [train_y2.shape[1]]))

network=VisNN.DrawNN(network_structure, clf.coefs_)
network.draw()

network_noweight=VisNN.DrawNN([8,9,1])
network_noweight.draw()