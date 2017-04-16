#N Dhanachandra Singh & Rimjhim
import dicom 
import os
import cv2
import numpy as np
import pandas as pd
import glob
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from scipy.misc import toimage
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,confusion_matrix
np.random.seed(7)

# DICOM rescale correction
def rescale_correction(temp):
	temp.image = temp.pixel_array * temp.RescaleSlope + temp.RescaleIntercept
	return temp
id_label_pairs = pd.read_csv('stage1_labels.csv')
ids = id_label_pairs['id'].values

dir = 'stage1'

images_path = []
for p in ids:	
    images_path += glob.glob("{}/{}/*.dcm".format(dir, p))
    #print images_path
print('Number of images: ',len(images_path))
data = []
label = []
for f in images_path:
	dico = dicom.read_file(f)
	slices = []
	dico=rescale_correction(dico)
	slices.append(dico)
	try:
		slices = sorted(slices, key=lambda x: x.SliceLocation)
	except:
		slices=slices[:len(slices)-1]
		continue
	pic = slices[ int(len(slices)/2) ].image.copy()
	#transforming images into grey scale
	pic[pic>-300] = 255
	pic[pic<-300] = 0
	pic = np.uint8(pic)
	#print pic
    	# find surrounding torso from the threshold and make a mask
	# finding boundary of the given image
    	im2, contours, _ = cv2.findContours(pic,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    	largest_contour = max(contours, key=cv2.contourArea)
    	mask = np.zeros(pic.shape, np.uint8)#create an array of zeroes
    	cv2.fillPoly(mask, [largest_contour], 255)# the boundary area(contour) is filed with zeroes
    	# apply mask to threshold image to remove outside. this is our new mask
	#print pic
	   	
	pic = ~pic
	#print pic
	#print type(pic)
    	pic[(mask == 0)] = 0 # <-- Larger than threshold value

    	# apply closing to the mask
    	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    	pic = cv2.morphologyEx(pic, cv2.MORPH_OPEN, kernel)  # <- to remove speckles...
    	pic = cv2.morphologyEx(pic, cv2.MORPH_DILATE, kernel)
    	pic = cv2.morphologyEx(pic, cv2.MORPH_DILATE, kernel)
    	pic = cv2.morphologyEx(pic, cv2.MORPH_CLOSE, kernel)
    	pic = cv2.morphologyEx(pic, cv2.MORPH_CLOSE, kernel)
    	pic = cv2.morphologyEx(pic, cv2.MORPH_ERODE, kernel)
    	pic = cv2.morphologyEx(pic, cv2.MORPH_ERODE, kernel)
    	
    	# apply mask to image
    	pic2 = slices[ int(len(slices)/2) ].image.copy()
    	pic2[(pic == 0)] = -2000 # <-- Larger than threshold value
		
	patient_id = os.path.basename(os.path.dirname(f))
        is_cancer = id_label_pairs.loc[id_label_pairs['id'] == patient_id]['cancer'].values[0]
	#if is_cancer == 1:
		#print (f)
	out=[]
	if is_cancer == 0:
                out = [0, 1]
        else:
                out = [1, 0]
	
	pic2 = cv2.resize(pic2, (100, 100), interpolation=cv2.INTER_CUBIC)
	
	data.append([pic2])
        label.append(out)
	#print pic2
split_value = int(round(0.8*len(data)))
data,label = shuffle(data,label, random_state=2)
X_train = data[:split_value]
Y_train = label[:split_value]
X_test = data[split_value:]
Y_test = label[split_value:]
#print ('Input shape: ', np.array(X_train).shape[0])
X_train = np.array(X_train).reshape(np.array(X_train).shape[0], 1, 100, 100)
X_test = np.array(X_test).reshape(np.array(X_test).shape[0], 1, 100, 100)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#print ('Input shape after reshape: ',X_train[0].shape)
#print (len(X_test))
#print (Y_test[:1])
#print (np.array(Y_train[0]).shape)
def myneuralnetwork():
	model = Sequential()
	#adding 0 padding of size 1*1
	model.add(ZeroPadding2D((1, 1), input_shape=(1, 100, 100), dim_ordering='tf'))
	#adding convolutional layer with filter of size 4
	model.add(Convolution2D(4, 1, 1, activation='relu'))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(4, 1, 1, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	'''
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(8, 1, 1, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(8, 1, 1, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	'''
	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	'''
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	'''
	model.add(Dense(2, activation='softmax'))
	sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=False)
	model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
	return model
model=myneuralnetwork()
#print(model.summary())
history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=100, batch_size=100)

# Final evaluation of the model

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predictions = model.predict(X_test, verbose=1, batch_size=100)
#X_test = np.array(X_test).reshape(np.array(X_test).shape[0], 100, 100)
#X_test = np.array(X_test).reshape(np.array(X_test).shape[0], 100, 100)

print ('dimension',np.array(predictions[0]))
print ('dimension',predictions.shape)
#plt.plot(X_test, predictions, 'ro')
#plt.show()

#confusion matrix

y_pred = model.predict_classes(X_test)
print('x_predict', y_pred[0])

p=model.predict_proba(X_test) # to predict probability

target_names = ['class 0(CANCER)', 'class 1(NON_CANCER)']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

scores = model.evaluate(X_test, Y_test, verbose=0)
print (scores)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
#print("Saved model to disk")
print (scores)
print("Accuracy: %.2f%%" % (scores[1]*100))
