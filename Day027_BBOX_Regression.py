import cv2
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt

##讀入照片
image=cv2.imread('data/Dog.JPG')

##因為CV2會將照片讀成BGR，要轉回來
image=image[:,:,::-1]

Ground_Truth_BBOX=[1900,700,1800,1800]
Region_Proposal=[1800,800,1500,1500]

plt.rcParams['figure.figsize'] = (20, 10)


fig,ax = plt.subplots(1)

##畫出圖片
ax.imshow(image)

# 畫BBOX-Prediction
rect = patches.Rectangle((Region_Proposal[0],Region_Proposal[1]),Region_Proposal[2],Region_Proposal[3],linewidth=3,edgecolor='r',facecolor='none',)
ax.text(1800,800,'Region_Proposal',size=20)
# 畫BBOX-Ground_Truth
rect_1 = patches.Rectangle((Ground_Truth_BBOX[0],Ground_Truth_BBOX[1]),Ground_Truth_BBOX[2],Ground_Truth_BBOX[3],linewidth=3,edgecolor='b',facecolor='none')
ax.text(1900,700,'Ground Truth',size=20)

# Add the patch to the Axes
ax.add_patch(rect)
ax.add_patch(rect_1)

plt.show()

tx=(Ground_Truth_BBOX[0]-Region_Proposal[0])/Region_Proposal[2]
ty=(Ground_Truth_BBOX[1]-Region_Proposal[1])/Region_Proposal[3]
tw=np.log(Ground_Truth_BBOX[2]/Region_Proposal[2])
th=np.log(Ground_Truth_BBOX[3]/Region_Proposal[3])

print('x偏移量： ',tx)
print('y偏移量： ',ty)
print('w縮放量： ',tw)
print('h縮放量： ',th)

dx,dy,dw,dh=[0.05,-0.05,0.12,0.17]

Loss=np.sum(np.square(np.array([tx, ty, tw, th])-np.array([dx, dy, dw, dh])))
# Loss=np.sum(np.square(np.array([tx,ty,tw,th])-np.array([0.05,-0.05,0.12,0.17])))

print('Loss值：',Loss)