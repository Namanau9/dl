import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data=[
"Hello","Hi","How are you?","I'm doing well, thanks!",
"What's your name?","I'm a chatbot.","Tell me a joke",
"Why don't scientists trust atoms?","Because they make up everything!","exit"]

t=Tokenizer()
t.fit_on_texts(data)
n=len(t.word_index)+1

seq=[]
for line in data:
    a=t.texts_to_sequences([line])[0]
    for i in range(1,len(a)):
        seq.append(a[:i+1])

L=max(map(len,seq))
seq=np.array(pad_sequences(seq,maxlen=L,padding='pre'))
X,y=seq[:,:-1],seq[:,-1]
y=tf.keras.utils.to_categorical(y,num_classes=n)

m=tf.keras.models.Sequential([
tf.keras.layers.Embedding(n,64,input_length=L-1),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
tf.keras.layers.Dense(n,activation='softmax')])

m.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
m.fit(X,y,epochs=100,verbose=1)

def response(x):
    a=t.texts_to_sequences([x])[0]
    a=pad_sequences([a],maxlen=L-1,padding='pre')
    i=np.argmax(m.predict(a)[0])
    return list(t.word_index.keys())[i-1] if 1<=i<=len(t.word_index) else 'UNKNOWN_WORD'

u='Hello'
while u!='exit':
    print(f"User: {u}")
    print(f"Chatbot: {response(u)}")
    u=input("You: ")
