# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:11:30 2020

@author: Jeremias Abreu
Email: jeremias10j@gmail.com
Github: j-abreu
"""


import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#%%
try:
    os.chdir(r'C:\Users\joyce\Desktop\Jeremie\github\tensorflow-examples')
    print('changing work directory...')
    print('current work directory %s' % (os.getcwd()))
except:
    print('current work directory %s' % (os.getcwd()))
    
#%%
np.random.seed(99)

print('\nTensorFlow version: {}'.format(tf.__version__))
print('Eager execution: {}'.format(tf.executing_eagerly()))

#%% loading and preparing data
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = K.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)
print("Local copy of the dataset file in: %s" % (train_dataset_fp))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32

train_dataset = pd.read_csv(train_dataset_fp, header=None)

train_dataset = np.array(train_dataset[1:], dtype='float32')

X_train = train_dataset[:,:-1]
y_train = train_dataset[:,-1]

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=len(X_train)).batch(batch_size)


#%% create model
def create_model():
    model = K.Sequential()
    
    model.add(K.layers.Dense(10, input_dim=4, activation='relu'))
    model.add(K.layers.Dense(10, activation='relu'))
    model.add(K.layers.Dense(3, activation='softmax'))
    
    model.summary()
    return model

#%% training and validation functions
def train_on_batch(X, y):
    with tf.GradientTape() as tape:
        yhat = model(X, training=True)
        loss_value = loss_object(y, yhat)
    
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value

def validate_on_batch(X, y):
    yhat = model(X, training=False)
    loss_value = loss_object(y, yhat)
    return loss_value

#%% define loss object and optimizer
loss_object = K.losses.sparse_categorical_crossentropy
optimizer = K.optimizers.Adam()

#%% training loop

model = create_model()

train_loss_history = []
train_acc_history = []

epochs = 200

for epoch in range(epochs):
    epoch_loss_avg = K.metrics.Mean()
    epoch_acc = K.metrics.SparseCategoricalAccuracy()
    
    # train for each batch
    for X, y in train_data:
        loss_value = train_on_batch(X, y)
        epoch_loss_avg.update_state(loss_value)
        epoch_acc.update_state(y, model(X, training=True))
    
    # save loss and accuracy for each epoch
    train_loss_history.append(epoch_loss_avg.result())
    train_acc_history.append(epoch_acc.result())
    
    if epoch % 50 == 0:
        print('Epoch {}\tLoss: {}\tAccuracy: {}'.format(epoch,
                                                        train_loss_history[-1],
                                                        train_acc_history[-1]))
    
#%% plot learning curves
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel('Loss', fontsize=14)
axes[0].plot(train_loss_history)

axes[1].set_ylabel('Accuracy', fontsize=14)
axes[1].set_xlabel('Epochs', fontsize=14)
axes[1].plot(train_acc_history)

plt.show()  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
