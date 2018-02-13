import os
import cv2

ht = 50
i = 0
os.chdir('/home/pallab/Pictures/Data/pos')
for pic in os.listdir('.'):
    img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
    height = img.shape[0]
    width = img.shape[1]
    size = (int(width*(ht/height)), ht)
    img = cv2.resize(img, size)
    cv2.imwrite('/home/pallab/Pictures/Data/pos50/p'+str(i)+'.jpg', img)
    i += 1

print('Resized', i, 'files', sep = ' ')
