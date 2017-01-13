import keras.backend as K
from roc_auc import RocAucScoreOp, PrecisionOp, RecallOp, F1Op
from roc_auc import AUCEpoch, PrecisionEpoch, RecallEpoch, F1Epoch, LossEpoch, ACCEpoch
from keras.preprocessing.image import ImageDataGenerator
#from image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, SpatialDropout2D
from keras.layers import advanced_activations
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1l2
#import googlenet
from convnetskeras.convnets import preprocess_image_batch, convnet
#import inception_v3
#import pickle
import cPickle as pickle
import os
import h5py
np.random.seed(1)
#srng = RandomStreams(1)
img_ext = '.pickle' # read data file and label
preprocesspath = '/preprocessedData/'
prefix = preprocesspath#'/scratch/'
img_fnames = [x for x in os.listdir(preprocesspath) if x.endswith('.h5')]
print(img_fnames)
img_fnames = [x for x in os.listdir(preprocesspath) if x.endswith(img_ext)]
print(img_fnames)
labelpath = '/preprocessedData/metadata/image_labels.txt'
with open(labelpath, 'r') as f:
  labellines = f.readlines()
#assert(len(labellines) == len(img_fnames))
fnamels = [0]*len(labellines)
labells = [0]*len(labellines)
count = 0
for ll in labellines:
  fname, label = ll.split()
  pckname = fname[:-4]+'227.pickle'
  if pckname not in img_fnames:
  	continue
  fnamels[count] = fname[:-4]+'227.pickle'
  labells[count] = int(label)
  assert(labells[count]==0 or labells[count]==1)
  count += 1
randindex = np.random.permutation(len(labells)) # shuffle the data
fnamels = np.array(fnamels)
fnamels = fnamels[randindex]
labells = np.array(labells)
labells = labells[randindex]
print('num images'+str(len(labells)))
valnum = min(100000, len(labells)//2)  # train val separation
print('val num'+str(valnum))
traindata = fnamels[valnum:]
trainlabel = labells[valnum:]
valdata = fnamels[:valnum]
vallabel = labells[:valnum]
ratio = sum(labells)*1.0 / len(labells)# imbalanced weights
weights = np.array((ratio, 1-ratio))
print('weights'+str(weights))
# parameters
lr = 5e-5
nb_epoch = 50
batch_size = 128
l2factor = 0#1e-5
l1factor = 0#2e-7
weighted = False #True #True
noises = 20 # 50
data_augmentation = True
modelname = 'alexnet' # miccai16, alexnet, levynet, googlenet
pretrain = True
mil=True
savename = '/modelState/'+modelname+'_lr'+str(lr)+'_l2'+str(l2factor)+'_l1'+str(l1factor)+\
'_ep'+str(nb_epoch)+'_bs'+str(batch_size)+'_w'+str(weighted)+str(noises)+str(pretrain)+'_mil'+str(mil)
print('savename'+savename)
nb_classes = 2
# input image dimensions
img_rows, img_cols = 227, 227
# the CIFAR10 images are RGB
img_channels = 3
trainbatch, trainlabelbatch = [], []
print('store into some big batches for train')
for i in xrange(0, len(traindata), batch_size*1000): # store into some big batches
  lenbatch = min(batch_size*1000, len(traindata)-i)
#  trainbatchdata = np.zeros((lenbatch, img_channels, img_rows, img_cols))
#  for j in xrange(i, i+lenbatch):
    #print j
#    with open(preprocesspath+traindata[j], 'rb') as f:
#      data = pickle.load(f)
#    data = data*1.0 / 255.0
#    trainbatchdata[j-i,0,:,:] =data
#    trainbatchdata[j-i,1,:,:] = data
#    trainbatchdata[j-i,2,:,:] = data 
#  h5f = h5py.File(prefix+str(i)+'.h5', 'w')
#  h5f.create_dataset(prefix+str(i)+'.h5', data=trainbatchdata)
#  h5f.close()
  #with open(prefix+str(i)+'.pickle', 'wb') as f:
  #  pickle.dump(trainbatchdata, f)
  trainbatch.append(prefix+str(i)+'.h5')
  trainlabelbatch.append(np_utils.to_categorical(trainlabel[i:i+lenbatch], nb_classes))
valbatch, vallabelbatch = [], []
print('store into some big batches for val')
for i in xrange(0, len(valdata), batch_size*1000): # store into some big batches
  assert(i==0)
  lenbatch = min(batch_size*1000, len(valdata)-i)
#  valbatchdata = np.zeros((lenbatch, img_channels, img_rows, img_cols))
#  for j in xrange(i, i+lenbatch):
#    with open(preprocesspath+valdata[j], 'rb') as f:
#      data = pickle.load(f)
#    data = data * 1.0 / 255.0
#    valbatchdata[j-i,0,:,:] =data
#    valbatchdata[j-i,1,:,:] = data
#    valbatchdata[j-i,2,:,:] = data 
#  h5f = h5py.File(prefix+str(i)+'val.h5', 'w')
#  h5f.create_dataset(prefix+str(i)+'val.h5', data=valbatchdata)
#  h5f.close()
  #with open(prefix+str(i)+'val.pickle', 'wb') as f:
  #  pickle.dump(valbatchdata, f)
  valbatch.append(prefix+str(i)+'val.h5')
  vallabelbatch.append(np_utils.to_categorical(vallabel[i:i+lenbatch], nb_classes))
h5f = h5py.File(valbatch[0], 'r')
valbatchdata = h5f[valbatch[0]][:]
h5f.close()
def rocauc(y_true, y_pred):
  auc = RocAucScoreOp()
  return auc(y_true, y_pred)
def precision(y_true, y_pred):
  prec = PrecisionOp()
  return prec(y_true, y_pred)
def recall(y_true, y_pred):
  reca = RecallOp()
  return reca(y_true, y_pred)
def f1(y_true, y_pred):
  f1 = F1Op()
  return f1(y_true, y_pred)

if modelname == 'alexnet':
  if pretrain:  # 227*227
    alexmodel = convnet('alexnet', weights_path='alexnet_weights.h5', heatmap=False, l1=l1factor, l2=l2factor)
    model = convnet('alexnet', outdim=2, l1=l1factor, l2=l2factor, usemil=mil)
    for layer, mylayer in zip(alexmodel.layers, model.layers):
      print(layer.name)
      if mylayer.name == 'mil_1':
        break
      else:
        weightsval = layer.get_weights()
        print(len(weightsval))
        mylayer.set_weights(weightsval)
  else:
    model = convnet('alexnet', outdim=2, l1=l1factor,l2=l2factor, usemil=mil)
elif modelname == 'VGG_16' or modelname == 'VGG_19':
  X_train_extend = np.zeros((X_train.shape[0],3, 224, 224))
  for i in xrange(X_train.shape[0]):
    rex = np.resize(X_train[i,:,:,:], (224, 224))
    X_train_extend[i,0,:,:] = rex
    X_train_extend[i,1,:,:] = rex
    X_train_extend[i,2,:,:] = rex
  X_train = X_train_extend
  X_test_extend = np.zeros((X_test.shape[0], 3,224, 224))
  for i in xrange(X_test.shape[0]):
    rex = np.resize(X_test[i,:,:,:], (224, 224))
    X_test_extend[i,0,:,:] = rex
    X_test_extend[i,1,:,:] = rex
    X_test_extend[i,2,:,:] = rex
  X_test = X_test_extend
  X_test_test_extend = np.zeros((X_test_test.shape[0], 3, 224, 224))
  for i in xrange(X_test_test.shape[0]):
    rex = np.resize(X_test_test[i,:,:,:], (224,224))
    X_test_test_extend[i,0,:,:] = rex
    X_test_test_extend[i,1,:,:] = rex
    X_test_test_extend[i,2,:,:] = rex
  X_test_test = X_test_test_extend
  if pretrain:  # 227*227
    if modelname == 'VGG_16':
      weightname = 'vgg16_weights.h5'
    else:
      weightname = 'vgg19_weights.h5'
    alexmodel = convnet(modelname, weights_path=weightname, heatmap=False, l1=l1factor, l2=l2factor)
    model = convnet(modelname, outdim=2, l1=l1factor, l2=l2factor, usemil=mil)
    for layer, mylayer in zip(alexmodel.layers, model.layers):
      print(layer.name)
      if mylayer.name == 'mil_1':
        break
      else:
        weightsval = layer.get_weights()
        print(len(weightsval))
        mylayer.set_weights(weightsval)
  else:
    model = convnet(modelname, outdim=2, l1=l1factor,l2=l2factor, usemil=mil)
elif modelname == 'googlenet':
  model = inception_v3.InceptionV3(include_top=True, outdim=2)
  if pretrain: 
    googlemodel = inception_v3.InceptionV3(include_top=True, weights='imagenet')
    for layer, mylayer in zip(googlemodel.layers, model.layers):
      if layer.name == 'predictions':
        break
      weightsval = layer.get_weights()
      mylayer.set_weights(weightsval)

# let's train the model using SGD + momentum (how original).
sgd = Adam(lr=lr) #SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])#, AUCEpoch,PrecisionEpoch,RecallEpoch,F1Epoch])
print(model.summary())
#filepath = savename+'-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5' #-{val_auc:.2f}-\
#{val_prec:.2f}-{val_reca:.2f}-{val_f1:.2f}.hdf5'
#checkpoint0 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
#checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint0 = LossEpoch(savename, validation_data=(valbatchdata, vallabelbatch[0]), interval=1)
checkpoint1 = ACCEpoch(savename, validation_data=(valbatchdata, vallabelbatch[0]), interval=1)
checkpoint2 = AUCEpoch(savename, validation_data=(valbatchdata, vallabelbatch[0]), interval=1)
#checkpoint3 = PrecisionEpoch(savename, validation_data=(X_test, Y_test), interval=1)
#checkpoint4 = RecallEpoch(savename, validation_data=(X_test, Y_test), interval=1)
checkpoint5 = F1Epoch(savename, validation_data=(valbatchdata, vallabelbatch[0]), interval=1)
#checkpoint2 = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
#checkpoint3 = ModelCheckpoint(filepath, monitor='val_prec', verbose=1, save_best_only=True, mode='max')
#checkpoint4 = ModelCheckpoint(filepath, monitor='val_reca', verbose=1, save_best_only=True, mode='max')
#checkpoint5 = ModelCheckpoint(filepath, monitor='val_f1', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint0, checkpoint1, checkpoint2, checkpoint5]
#callbacks_list = [AUCEpoch, PrecisionEpoch, RecallEpoch, F1Epoch, checkpoint0, checkpoint1]

if not data_augmentation:
  print('Not using data augmentation.')
  if weighted:
    for e in range(nb_epoch):
      print("epoch %d" % e)
      for X_train, Y_train in zip(trainbatch, trainlabelbatch): # these are chunks of ~10k pictures
        h5f = h5py.File(X_train, 'r')
        X_traindata = h5f[X_train][:]
        h5f.close()
        #with open(X_train, 'rb') as f:
        #  X_traindata = pickle.load(f)
        #for X_batch, Y_batch in datagen.flow(X_traindata, Y_train, batch_size=batch_size): # these are chunks of 32 samples
        #    loss = model.train(X_batch, Y_batch)
        model.fit(X_traindata, Y_train,
                        batch_size=batch_size,
                        samples_per_epoch=X_traindata.shape[0],
                        nb_epoch=1,
                        validation_data=(valbatchdata, vallabelbatch[0]),
                        callbacks=callbacks_list,
                        class_weight=[weights[0], weights[1]])
  else:
    for e in range(nb_epoch):
      print("epoch %d" % e)
      for X_train, Y_train in zip(trainbatch, trainlabelbatch): # these are chunks of ~10k pictures
        h5f = h5py.File(X_train, 'r')
        X_traindata = h5f[X_train][:]
        h5f.close()
        #with open(X_train, 'rb') as f:
        #  X_traindata = pickle.load(f)
        #for X_batch, Y_batch in datagen.flow(X_traindata, Y_train, batch_size=batch_size): # these are chunks of 32 samples
        #    loss = model.train(X_batch, Y_batch)
        model.fit(X_traindata, Y_train,
                        batch_size=batch_size, 
                        samples_per_epoch=X_traindata.shape[0],
                        nb_epoch=1,
                        validation_data=(valbatchdata, vallabelbatch[0]),
                        callbacks=callbacks_list)
else:
  print('Using real-time data augmentation.')
  # this will do preprocessing and realtime data augmentation
  datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=45.0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)
#        zerosquare=True,
#        zerosquareh=noises,
#        zerosquarew=noises,
#        zerosquareintern=0.0)  # randomly flip images
  # compute quantities required for featurewise normalization
  # (std, mean, and principal components if ZCA whitening is applied)
  h5f = h5py.File(prefix+'0.h5', 'r')
  X_traindata = h5f[prefix+'0.h5'][:]
  h5f.close()
  #with open(prefix+'0.pickle', 'rb') as f:
  #  X_traindata = pickle.load(f)
  datagen.fit(X_traindata)
  # fit the model on the batches generated by datagen.flow()
  if weighted:
    for e in range(nb_epoch):
      print("epoch %d" % e)
      for X_train, Y_train in zip(trainbatch, trainlabelbatch): # these are chunks of ~10k pictures
        h5f = h5py.File(X_train, 'r')
        X_traindata = h5f[X_train][:]
        h5f.close()
        #with open(X_train, 'rb') as f:
        #  X_traindata = pickle.load(f)
        #for X_batch, Y_batch in datagen.flow(X_traindata, Y_train, batch_size=batch_size): # these are chunks of 32 samples
        #    loss = model.train(X_batch, Y_batch)
        model.fit_generator(datagen.flow(X_traindata, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_traindata.shape[0],
                        nb_epoch=1,
                        validation_data=(valbatchdata, vallabelbatch[0]),
                        callbacks=callbacks_list,
                        class_weight=[weights[0], weights[1]])
  else:
    for e in range(nb_epoch):
      print("epoch %d" % e)
      for X_train, Y_train in zip(trainbatch, trainlabelbatch): # these are chunks of ~10k pictures
        h5f = h5py.File(X_train, 'r')
        X_traindata = h5f[X_train][:]
        h5f.close()
        #with open(X_train, 'rb') as f:
        #  X_traindata = pickle.load(f)
        #for X_batch, Y_batch in datagen.flow(X_traindata, Y_train, batch_size=batch_size): # these are chunks of 32 samples
        #    loss = model.train(X_batch, Y_batch)
        model.fit_generator(datagen.flow(X_traindata, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_traindata.shape[0],
                        nb_epoch=1,
                        validation_data=(valbatchdata, vallabelbatch[0]),
                        callbacks=callbacks_list)