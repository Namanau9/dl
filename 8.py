from keras.datasets import imdb
from keras import Sequential
from keras.layers import Embedding,Dense,SimpleRNN
from keras.preprocessing import sequence

vocabulary_size=5000
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=vocabulary_size)

print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train),len(X_test)))
print('---review---')
print(X_train[6])
print('---label---')
print(y_train[6])

word2id=imdb.get_word_index()
id2word={i:word for word,i in word2id.items()}

print('---review with words---')
print([id2word.get(i,' ') for i in X_train[6]])
print('---label---')
print(y_train[6])

print('Maximum review length: {}'.format(len(max(X_train+X_test,key=len))))
print('Minimum review length: {}'.format(len(min(X_test+X_test,key=len))))

max_words=500
X_train=sequence.pad_sequences(X_train,maxlen=max_words)
X_test=sequence.pad_sequences(X_test,maxlen=max_words)

m=Sequential()
m.add(Embedding(vocabulary_size,32,input_length=max_words))
m.add(SimpleRNN(100))
m.add(Dense(1,activation='sigmoid'))

m.summary()
m.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

X_valid,y_valid=X_train[:64],y_train[:64]
X_train2,y_train2=X_train[64:],y_train[64:]

m.fit(
X_train2,y_train2,
validation_data=(X_valid,y_valid),
batch_size=64,
epochs=1)

scores=m.evaluate(X_test,y_test,verbose=0)
print('Test accuracy:',scores[1])
