import tensorflow as tf
from tensorflow import keras

(x_train,y_train),(x_test,y_test)=keras.datasets.cifar10.load_data()
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

m=keras.Sequential([
keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)),
keras.layers.MaxPooling2D((2,2)),
keras.layers.Flatten(),
keras.layers.Dense(128,activation="relu"),
keras.layers.Dense(10,activation="softmax")])

m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
m.fit(x_train,y_train,epochs=10,batch_size=64,validation_split=0.2)
test_loss,test_accuracy=m.evaluate(x_test,y_test)
print(f"Test accuracy: {test_accuracy*100:.2f}%")
