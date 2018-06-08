

"""
@author: rabiaabuaqel
"""
################################################
#IMPORTS
################################################
from keras.applications.resnet50 import ResNet50
from sklearn.cross_validation import train_test_split
import os
import numpy as np
from time import time
from PIL import Image
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math
################################################
#VARS
################################################
DATA_DIR = '/data'
X = []
Y = np.zeros(1907)
SIZE = (224, 224)
BATCH_SIZE = 64
data = os.listdir(DATA_DIR)
data.remove('.DS_Store')
data.remove('.ipynb_checkpoints')
data.sort(key = lambda x: int(x))
max_index = 0
min_index = 0
################################################
#LOAD DATA
################################################
for sub in data:
    
    sub_images = os.listdir(DATA_DIR +"/" + sub)
    if '.DS_Store' in sub_images:
        sub_images.remove('.DS_Store')
    sub_images.sort(key = lambda x:int(x.split(".")[0]))
    max_index = min_index + len(sub_images)
    Y[min_index:max_index] = sub
    min_index = max_index
    
    for img in sub_images:
        X.append(np.asarray(Image.open(DATA_DIR +"/" + sub + "/" + img)))
            
   
    
np.set_printoptions(threshold=np.inf)
X = np.array(X)
################################################
#SPLIT TRAIN-VAL-TEST
################################################
X_train, X_test, y_train, y_test = \
train_test_split(X, Y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = \
train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    
#print (y_train.shape)
num_valid_steps = math.floor(X_val.shape[0]/BATCH_SIZE)
################################################
#LOAD MODEL
################################################
model = ResNet50(weights='imagenet')
# remove output layer of the model
model.layers.pop()
# Turn of training for all network layers
for layer in model.layers:
    layer.trainable=False
    
# Add Softmax regression for output layer    
last = model.layers[-1].output
x = Dense(len(data), activation="softmax")(last)
finetuned_model = Model(model.input, x)
finetuned_model.compile(optimizer=Adam(lr=0.0001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
finetuned_model.classes = y_train
early_stopping = EarlyStopping(patience=10)
checkpointer = ModelCheckpoint('resnet50_best.h5',
                               verbose=1, save_best_only=True)
################################################
#TRAINING
################################################
t0 = time()
history = finetuned_model.fit(X_train, y_train, batch_size=16,
                              epochs=16, verbose=1,
                    callbacks=[early_stopping, checkpointer],
                    validation_split=0.0,
                    validation_data=(X_val,y_val), shuffle=True,
                    class_weight=None, sample_weight=None,
                    initial_epoch=0)
print("Model Training Time: %0.3fs" % (time() - t0))
finetuned_model.save('resnet50_final.h5')
################################################
#EVALUATION
################################################
t0 = time()
score = finetuned_model.evaluate(X_test, y_test,
                                 batch_size=16, verbose=1)
print("Model Evaluation Time: %0.3fs" % (time() - t0))                               
print ("Model Evaluation Score:")
print (score)
