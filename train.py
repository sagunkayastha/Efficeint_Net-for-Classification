from time import time

from keras import applications
from efficientnet import EfficientNetB3
from keras import callbacks
from keras.models import Sequential


from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from utils import PolynomialDecay,get_lr_metric
from keras.callbacks import TensorBoard
import argparse

class Efficient:

    def __init__(self,train_dir,test_dir,batch_size,epochs=100,effN='3'):
        self.batch_size = batch_size
        self.epochs = epochs    
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        self.effN=effN
        self.test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = self.train_datagen.flow_from_directory(
            train_dir,
            target_size=(32,32),
            batch_size=self.batch_size,
            class_mode='categorical')

        self.test_generator = self.test_datagen.flow_from_directory(
            test_dir,
            target_size=(32, 32),
            batch_size=self.batch_size,
            class_mode='categorical')

        

        if self.effN=='3':
            self.efficient_net = EfficientNetB3(
                
                input_shape=(32,32,3),
                include_top=False,
                pooling='max'
            )
            print('Using Model EfficientNetB3')

        elif self.effN=='2':
            from efficientnet import EfficientNetB2
            self.efficient_net = EfficientNetB2(
                
                input_shape=(32,32,3),
                include_top=False,
                pooling='max'
            )
            print('Using Model EfficientNetB2')

        elif self.effN =='4':
            from efficientnet import EfficientNetB4
            self.efficient_net = EfficientNetB4(
                
                input_shape=(32,32,3),
                include_top=False,
                pooling='max'
            )
            print('Using Model EfficientNetB4')

        elif self.effN=='5':
            from efficientnet import EfficientNetB5
            self.efficient_net = EfficientNetB5(
                
                input_shape=(32,32,3),
                include_top=False,
                pooling='max'
            )
            print('Using Model EfficientNetB5')


    def efficient_model(self):
        self.model = Sequential()
        self.model.add(self.efficient_net)
        self.model.add(Dense(units = 120, activation='relu'))
        self.model.add(Dense(units = 120, activation = 'relu'))
        self.model.add(Dense(units = 23, activation='sigmoid'))
        self.model.summary()	

        optimizer=Adam(lr=0.01)
        lr_metric = get_lr_metric(optimizer)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy',lr_metric])
        filepath="checkpoints/weights.hdf5"

        schedule = PolynomialDecay(maxEpochs=self.epochs, initAlpha=1e-1, power=5)
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [checkpoint,tensorboard]
        #self.callbacks_list = [checkpoint,LearningRateScheduler(schedule),tensorboard]

    def train_model(self):
        self.efficient_model()
        history = self.model.fit_generator(
            self.train_generator,
            epochs = 100,
            steps_per_epoch=len(self.train_generator),
            validation_data=self.test_generator,
            validation_steps=len(self.test_generator),
            callbacks=self.callbacks_list
    )



parser = argparse.ArgumentParser(description='Efficient')
parser.add_argument('-train_dir', type=str, help='Train_dir',default = '../dataset2/train/')
parser.add_argument('-test_dir', type=str, help='Train_dir', default = '../dataset2/test')
parser.add_argument('-batch_size', type=int, help='batch_size', default = 32)
parser.add_argument('-epochs', type=int, help='epoch', default = 100)
parser.add_argument('-model', type=str, help='Model Number', default = '3')
args = parser.parse_args()
# train_dir = '../dataset2/train/'
# test_dir = '../dataset2/test'
# print(args.train_dir)
# exit()
Network = Efficient(args.train_dir,args.test_dir,args.batch_size, args.epochs, args.model)
Network.train_model()

#python.exe .\train.py ../dataset2/train/ ../dataset2/test 32 100 3
#tensorboard --logdir=logs/ -port 5252