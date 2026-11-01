from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10

(xtr,ytr),(xte,yte)=cifar10.load_data()
xtr=xtr.reshape(len(xtr),-1)/255
xte=xte.reshape(len(xte),-1)/255

m=KNeighborsClassifier(3)
m.fit(xtr[:5000],ytr[:5000].ravel())
p=m.predict(xte[:1000])
print(f"KNN Test Accuracy (on 1000 samples): {accuracy_score(yte[:1000],p):.4f}")
