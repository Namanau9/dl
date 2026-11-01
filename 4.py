import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()

X_train=X_train/255
X_test=X_test/255
Y_train=to_categorical(Y_train,10)
Y_test=to_categorical(Y_test,10)

d=ImageDataGenerator(
rotation_range=10,
horizontal_flip=True,
width_shift_range=0.1,
height_shift_range=0.1)

g=d.flow(X_train,Y_train,batch_size=64)

m=Sequential()
m.add(Conv2D(32,(4,4),input_shape=(32,32,3),activation='relu'))
m.add(MaxPooling2D(pool_size=(2,2)))
m.add(Conv2D(32,(4,4),input_shape=(32,32,3),activation='relu'))
m.add(MaxPooling2D(pool_size=(2,2)))
m.add(Flatten())
m.add(Dense(128,activation='relu'))
m.add(Dense(10,activation='softmax'))

m.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
m.summary()
m.fit(g,epochs=10,verbose=1,validation_data=(X_test,Y_test))
test_loss,test_accuracy=m.evaluate(X_test,Y_test)
print(f"Test accuracy: {test_accuracy*100:.2f}%")
