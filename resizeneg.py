''' IMPORTANT: Run this .py file from the terminal ONLY after navigating to the folder where your raw images are located
	This script will read images from the current directory and resize them to 100x100, in grayscale, and put them
	in the mentioned folder

'''

import os
import cv2

ht = 100
i = 0
os.chdir('/home/pallab/Pictures/Data/rawimgs')
for pic in os.listdir('.'):
    img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (ht, ht))
    cv2.imwrite('/home/pallab/Pictures/Data/treatedimgs/'+str(i)+'.jpg', img)
    i += 1

print('Resized and greyed', i, 'files', sep = ' ')
