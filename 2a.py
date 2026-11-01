import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image

m=MobileNetV2(weights='imagenet')
x=image.img_to_array(image.load_img('your_image.jpg',target_size=(224,224)))
x=preprocess_input(np.expand_dims(x,0))
p=m.predict(x)
for i,(_,label,prob) in enumerate(decode_predictions(p,top=3)[0]):
    print(f"{i+1}. {label}: {prob*100:.2f}%")
