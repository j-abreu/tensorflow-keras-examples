# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:11:30 2020

@author: Jeremias Abreu
Email: jeremias10j@gmail.com
Github: j-abreu
"""

def main():
    import tensorflow as tf
    import tensorflow.keras as K
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    #%%
    try:
        os.chdir(r'C:\Users\joyce\Desktop\Jeremie\github\tensorflow-keras-examples')
        print('changing work directory...')
        print('current work directory %s' % (os.getcwd()))
    except:
        print('current work directory %s' % (os.getcwd()))
        
    #%%
    np.random.seed(99)
    
    print('\nTensorFlow version: {}'.format(tf.__version__))
    print('Eager execution: {}'.format(tf.executing_eagerly()))
    
    #%% loading and preparing data
    
    # download training dataset
    train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
    train_dataset_fp = K.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)
    print("Local copy of the training dataset file in: %s" % (train_dataset_fp))
    
    # download testing dataset
    test_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
    test_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(test_dataset_url), origin=test_dataset_url)
    print('Locas copy of the testing dataset file in: %s' % (test_dataset_fp))
    
    
    class_names = {0:'Iris setosa', 1:'Iris versicolor', 2:'Iris virginica'}
    
    #read test and training dataset
    train_dataset = pd.read_csv(train_dataset_fp, header=None)
    test_dataset = pd.read_csv(test_dataset_fp, header=None)
    
    train_dataset = np.array(train_dataset[1:], dtype='float32')
    test_dataset = np.array(test_dataset[1:], dtype='float32')
    
    # separate features from labels 
    X_train = train_dataset[:,:-1]
    y_train = train_dataset[:,-1]
    
    X_test = test_dataset[:,:-1]
    y_test = test_dataset[:,-1]
    
    
    # build dataset object with data shuffled and batch of length 32
    batch_size = 32
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=len(X_train)).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(buffer_size=len(X_test)).batch(batch_size)
    
    train_data_total_len = np.sum([len(x) for x, _ in train_data])
    test_data_total_len = np.sum([len(x) for x, _ in test_data])
    train_num_batchs = len(list(train_data))
    test_num_batchs = len(list(test_data))
    
    
    #%% create model
    def create_model():
        model = K.Sequential()
        
        model.add(K.layers.Dense(10, input_dim=4, activation='relu'))
        model.add(K.layers.Dense(10, activation='relu'))
        model.add(K.layers.Dense(3, activation='softmax'))
        
        model.summary()
        return model
    
    #%% training and validation functions
    @tf.function
    def train_on_batch(X, y):
        with tf.GradientTape() as tape:
            yhat = model(X, training=True)
            loss_value = loss_object(y, yhat)
        
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value, yhat
    
    @tf.function
    def validate_on_batch(X, y):
        yhat = model(X, training=False)
        loss_value = loss_object(y, yhat)
        return loss_value, yhat
    
    def verbose(cur_epoch, epochs, data_processed, data_total_len, cur_batch, num_batchs):
        bar_len = 30
        progress_bar = int(data_processed/data_total_len*bar_len)
        print('\r%s' % (' '*50), end='')
        print('\rEpoch [%d/%d] [%s%s] bacth [%d/%d]' % (cur_epoch, epochs,
                                                        '-'*progress_bar, ' '*(bar_len-progress_bar),
                                                        cur_batch, num_batchs), end='')
        
    
    #%% define loss object and optimizer
    loss_object = K.losses.sparse_categorical_crossentropy
    optimizer = K.optimizers.Adam()
    
    model = create_model()
    
    #%% training loop
    
    train_loss_history = []
    train_acc_history = []
    
    test_loss_history = []
    test_acc_history = []
    
    epochs = 300
    
    for epoch in range(epochs):
        epoch_train_loss_avg = K.metrics.Mean()
        epoch_train_acc = K.metrics.SparseCategoricalAccuracy()
        
        epoch_test_loss_avg = K.metrics.Mean()
        epoch_test_acc = K.metrics.SparseCategoricalAccuracy()
        
        # train for each batch
        processed_data = 0
        for batch, (X, y) in enumerate(train_data):
            processed_data += len(X)
            (loss_value, predictions) = train_on_batch(X, y)
            epoch_train_loss_avg.update_state(loss_value)
            epoch_train_acc.update_state(y, predictions)
            verbose(epoch+1, epochs, processed_data, train_data_total_len, batch+1, train_num_batchs)
            
        # save training loss and accuracy for each epoch
        train_loss_history.append(epoch_train_loss_avg.result())
        train_acc_history.append(epoch_train_acc.result())
        print('\ntrain_loss: {}\ttrain_acc: {}'.format(train_loss_history[-1],
                                                       train_acc_history[-1]), end='\t')
        
        for batch, (X, y) in enumerate(test_data):
            (loss_value, predictions) = validate_on_batch(X, y)
            epoch_test_loss_avg.update_state(loss_value)
            epoch_test_acc.update_state(y, predictions)
            
        # save test loss and accuracy for each epoch
        test_loss_history.append(epoch_test_loss_avg.result())
        test_acc_history.append(epoch_test_acc.result())
        
        print('test_loss: {}\ttest_acc: {}'.format(test_loss_history[-1],
                                                   test_acc_history[-1]))
        print('\n')
    #%% plot learning curves
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')
    
    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].plot(train_loss_history, label='train loss')
    axes[0].plot(test_loss_history, label='test loss')
    axes[0].legend(loc='best')
    
    axes[1].set_ylabel('Accuracy', fontsize=14)
    axes[1].set_xlabel('Epochs', fontsize=14)
    axes[1].plot(train_acc_history, label='train acc')
    axes[1].plot(test_acc_history, label='test acc')
    axes[1].legend(loc='best')
    
    plt.show()  
    
if __name__ == '__main__':
    main()
