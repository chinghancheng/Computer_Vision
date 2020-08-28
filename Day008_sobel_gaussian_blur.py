import cv2
import numpy as np

img = cv2.imread('data/lena.png')

img_blur = img.copy()

# 重複多次 Gaussian 模糊的操作來加深模糊的程度
img_blur = cv2.GaussianBlur(img_blur, (5, 5), 0)
img_blur = cv2.GaussianBlur(img_blur, (5, 5), 0)
img_blur = cv2.GaussianBlur(img_blur, (5, 5), 0)

#　組合 + 顯示圖片
img_show = np.hstack((img, img_blur))
while True:
    cv2.imshow('Gaussian Blur', img_show)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 對 x 方向做 Sobel 邊緣檢測
img_sobel_x = cv2.Sobel(img_grey, cv2.CV_16S, dx=1, dy=0, ksize=3)
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

# 對 y 方向做 Sobel 邊緣檢測
img_sobel_y = cv2.Sobel(img_grey, cv2.CV_16S, dx=0, dy=1, ksize=3)
img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

# x, y 方向的邊緣檢測後的圖各以一半的全重進行合成
img_sobel_combine = cv2.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5, 0)

#　組合 + 顯示圖片
img_show = np.hstack((img_sobel_x, img_sobel_y, img_sobel_combine))
while True:
    cv2.imshow('Edge Detection_0', img_show)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break


# # 對 x 方向直接以非負整數的資料格式 (uint8) 進行 Sobel 邊緣檢測
# img_sobel_x_uint8 = cv2.Sobel(img_grey, cv2.CV_8S, dx=1, dy=0, ksize=3)

# img_sobel_x_uint8 = cv2.Sobel(img_sobel_x_uint8)
# #　組合 + 顯示圖片
# img_show = np.hstack((img_grey, img_sobel_x, img_sobel_x_uint8))
# while True:
#     # 比較 Sobel 邊緣檢測的過程中針對負數操作的不同產生的差異
#     cv2.imshow('Edge Detection_1', img_show)
#     k = cv2.waitKey(0)
#     if k == 27:
#         cv2.destroyAllWindows()
#         break


img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 求一次導數取得邊緣檢測結果
img_sobel_x = cv2.Sobel(img_grey, cv2.CV_16S, dx=1, dy=0)
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
# 求二次導數取得邊緣檢測結果
img_sobel_xx = cv2.Sobel(img_grey, cv2.CV_16S, dx=2, dy=0)
img_sobel_xx = cv2.convertScaleAbs(img_sobel_xx)
#　組合 + 顯示圖片
img_show = np.hstack((img_grey, img_sobel_x, img_sobel_xx))
while True:
    cv2.imshow('Edge Detection_2', img_show)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break