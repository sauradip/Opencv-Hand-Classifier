import urllib 
import cv2
import numpy as np
import os

def store_raw_images():
    neg_images_link = '//image-net.org/synset?wnid=n07942152&username=einsamerwolf&accesskey=4b4a62d0ac3bce512d39183dae14bff39cca4203'   
    neg_image_urls = urllib.urlopen(neg_images_link).read().decode()
    pic_num = 1
    
    if not os.path.exists('neg'):
        os.makedirs('neg')
        
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.urlretrieve(i, "neg/"+str(pic_num)+".jpg")
            img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e))

store_raw_images()
