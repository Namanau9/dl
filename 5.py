from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T

m=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
m.eval()

names=['background','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','N/A','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','N/A','backpack','umbrella','N/A','N/A','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','N/A','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','N/A','dining table','N/A','N/A','toilet','N/A','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','N/A','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

path='object_detection.jpg'
x=T.ToTensor()(Image.open(path))

with torch.no_grad():
    p=m([x])[0]

im=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

for b,l,s in zip(p['boxes'],p['labels'],p['scores']):
    if s>=0.8 and names[l] in ['dog','cat']:
        x1,y1,x2,y2=map(int,b)
        cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(im,names[l],(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

plt.imshow(im)
plt.show()
