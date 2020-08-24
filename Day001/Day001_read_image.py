import cv2
import os

init_dir = os.path.dirname(__file__)
img_path = os.path.join(init_dir, 'data/lena.png')
# 以彩色圖片的方式載入
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

# 以灰階圖片的方式載入
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 為了要不斷顯示圖片，所以使用一個迴圈
while True:
    # 顯示彩圖
    cv2.imshow('bgr', img)
    # 顯示灰圖
    # cv2.imshow('gray', img_gray)
    # 顯示
    cv2.imshow('r', img[:,:,0])
    cv2.imshow('g', img[:,:,1])
    cv2.imshow('b', img[:,:,2])

    # 直到按下 ESC 鍵才會自動關閉視窗結束程式
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break
