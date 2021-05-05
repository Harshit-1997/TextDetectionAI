import numpy as np  
from tensorflow.keras.layers import Conv2D,MaxPool2D,Input
from tensorflow.keras.models import Model
import tensorflow as tf

imgs = np.load("./MajorData/09AZ_space__binary_images.npy")
labels = np.load("./MajorData/09AZ_space__binary_labels.npy")

ids = list(range(imgs.shape[0]))
np.random.shuffle(ids)
imgs = imgs[ids]
labels = labels[ids]

def arch():
    X_in = Input((None,None,1))
    X = Conv2D(8,(3,3),activation='elu')(X_in)
    X = Conv2D(16,(3,3),activation='elu')(X)
    X = Conv2D(32,(3,3),activation='elu')(X)
    X = Conv2D(64,(3,3),activation='elu')(X)
    X = MaxPool2D()(X)
    X = Conv2D(64,(3,3),activation='elu')(X)
    X = Conv2D(32,(3,3),activation='elu')(X)
    X = Conv2D(16,(3,3),activation='elu')(X)
    X = Conv2D(8,(3,3),activation='elu')(X)
    X = MaxPool2D()(X)
    #X = Conv2D(800,(5,5),activation='relu')(X)
    #X = Conv2D(800,(1,1),activation='relu')(X)
    X_out = Conv2D(2,(1,1),activation='softmax')(X)
    return Model(inputs=X_in,outputs=X_out)

model = arch()
model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(imgs,labels,batch_size=128,epochs=1,validation_split=0.2,shuffle=True)

q = input("save?<y/n> : ")
if q=='y':
    tf.keras.models.save_model(model,"localization_model.hdf5")
