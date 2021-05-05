from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, Input
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Lambda
from tensorflow.keras.layers import Dense, LSTM, Reshape, Add
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode, get_value
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

import numpy as np

x_train = np.load("../MajorData/09AZ_images.npy")
y_train = np.load("../MajorData/09AZ_labels.npy")
x_test = np.load("../MajorData/09AZ_images.npy")
y_test = np.load("../MajorData/09AZ_labels.npy")


max_label_len=5

class DataGen(tf.keras.utils.Sequence):
    global max_label_len
    def __init__(self,batch_size,step_size,X,Y):
        self.batch_size=batch_size
        self.step_size=step_size
        self.X=X
        self.Y= Y
        self.on_epoch_end()
        print(Y.shape)
    
    def __len__(self):
        return self.step_size

    def on_epoch_end(self):
        #self.batch_size*=2
        #self.step_size//=2
        pass

    def __getitem__(self,index):
        x=self.X
        y=self.Y
        y=y[:,np.newaxis]
        l=np.random.randint(max_label_len-4,max_label_len+1)
        rx=np.empty((self.batch_size,28,28*l))
        ry=np.zeros((self.batch_size,max_label_len,1))

        for i in range(self.batch_size):
            j = np.random.randint(0,len(x),l)
            rx[i,]=np.concatenate([x[v] for v in j],axis=1)
            ry[i,:l]=np.concatenate([y[v] for v in j],axis=0)

        true_len = np.ones((self.batch_size,1))*l
        pred_len = np.ones((self.batch_size,1))*((28//8)*(28*l//8))
        inputs = {'X_data':rx[:,:,:,np.newaxis],
                  'labels':ry.reshape(self.batch_size,max_label_len),
                  'true_len':true_len,
                  'pred_len':pred_len}

        return inputs,np.zeros((self.batch_size,1))


def ctc_loss(args):
    y_true, y_pred, true_len, pred_len=args
    #input_len=np.zeros((y_pred.shape[0],1))
    #input_len[:,0]=y_pred.shape[1]
    return ctc_batch_cost(y_true, y_pred, pred_len, true_len)

def conv_33(X, channels,act):
    #X = Conv2D(channels,(1,3),padding='same',activation=act)(X)
    #X = Conv2D(channels,(3,1),padding='same',activation=act)(X)
    X = Conv2D(channels,(3,3),padding='same',activation=act)(X)
    return X

def conv_55(X, channels,act):
    #X = conv_33(X,channels,act)
    #X = conv_33(X,channels,act)
    X = Conv2D(channels,(5,5),padding='same',activation=act)(X)
    return X

def conv_77(X, channels,act):
    #X = conv_33(X,channels,act)
    #X = conv_33(X,channels,act)
    #X = conv_33(X,channels,act)
    X = Conv2D(channels,(7,7),padding='same',activation=act)(X)
    return X

def incep_layer(X, channels,act):
    X_11 = Conv2D(channels//4,(1,1),activation=act)(X)
    X_11 = MaxPool2D()(X_11)

    X_33 = conv_33(X, channels//4,act)
    X_33 = MaxPool2D()(X_33)

    X_55 = conv_55(X, channels//4,act)
    X_55 = MaxPool2D()(X_55)
    
    X_77 = conv_77(X, channels//4,act)
    X_77 = MaxPool2D()(X_77)

    X_m = MaxPool2D()(X)

    X = Concatenate(axis=3)([X_11,X_33,X_55,X_77,X_m])

    return X

def ctc_model():
    global max_label_len
    X_in = Input(shape=(None,None,1), name= 'X_data')

    X = conv_33(X_in,16,'elu')

    X = conv_33(X,32,'elu')

    X = incep_layer(X,32,'elu')

    X = conv_33(X,64,'elu')

    X = MaxPool2D()(X)

    X = conv_33(X,32,'elu')

    X = incep_layer(X,32,'elu')

    X = conv_33(X,32,'elu')

    #X = MaxPool2D()(X)

    X = conv_33(X,16,'elu')

    ##X = Conv2D(32,(3,3),padding='same',activation='elu')(X_in)
    ##X = MaxPool2D()(X)

    ##X = Conv2D(64,(3,3),padding='same',activation='elu')(X)
    ##X = MaxPool2D()(X)

    ##X = Conv2D(32,(3,3),padding='same',activation='elu')(X)
    ##X = MaxPool2D()(X)

    ##X = Conv2D(16,(3,3),padding='same',activation='elu')(X)
    ##X = MaxPool2D()(X)

    X = Reshape((-1,16))(X)

    ##X = Bidirectional(LSTM(32,return_sequences=True))(X)

    ##X = Bidirectional(LSTM(64,return_sequences=True))(X)

    X = Bidirectional(LSTM(32,return_sequences=True))(X)

    ##X = TimeDistributed(Dense(36,'elu'))(X)

    X = TimeDistributed(Dense(36,'elu'))(X)

    y_pred = TimeDistributed(Dense(36,'softmax'),name='output')(X)

    y_true = Input(shape=[max_label_len],name = 'labels')

    true_len = Input([1],name = 'true_len')

    pred_len = Input([1],name = 'pred_len')

    loss = Lambda(ctc_loss,name = 'ctc')([y_true, y_pred, true_len, pred_len])

    md = Model(inputs=[X_in,y_true,true_len,pred_len],outputs=loss)
    return md

lrs=LearningRateScheduler(lambda e:0.001-((e//10)*0.0002),verbose=1)

model = ctc_model()
adam=Adam(0.002)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer='adam',metrics=['acc'])

train_gen=DataGen(32,250,x_train,y_train)
test_gen=DataGen(32,5,x_test,y_test)

model.fit_generator(train_gen,epochs=50,validation_data=test_gen,callbacks=[lrs])

q = input("save?<y/n> : ")
if q=='y':
    save_model(model,"converter_model.hdf5")
