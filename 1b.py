import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.reshape((-1,32*32*3)).astype("float32")/255.0
x_test=x_test.reshape((-1,32*32*3)).astype("float32")/255.0
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

m=models.Sequential([
layers.Dense(512,activation='relu',input_shape=(32*32*3,)),
layers.Dense(256,activation='relu'),
layers.Dense(10,activation='softmax')])

m.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
m.fit(x_train,y_train,epochs=10,batch_size=64,validation_split=0.1)
loss,acc=m.evaluate(x_test,y_test)
print(f"MLP Test Accuracy: {acc:.4f}")
