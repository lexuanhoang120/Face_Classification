from genericpath import isdir
import cv2 
import numpy as np
import random as r

import os
# import torchvision.transforms.functional as F
def rotate_image(image):
    angle = r.randint(-10,10)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def change_brightness(img):
    bright_level = r.randint(-50,50)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,bright_level)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def crop_img(img):
    x1, y1,x2,y2 = r.randint(0,10), r.randint(0,10),r.randint(1,10), r.randint(1,10)
    img = img[x1:-x2,y1:-y2]
    
    return img
def flip_img(img):
    img = cv2.flip(img,1)
    return img

def augmentation_img(img):
    if r.randint(0,1):
        img = rotate_image(img)
    if r.randint(0,1):
        img = change_brightness(img)
    if r.randint(0,1):
        img = crop_img(img)
    if r.randint(0,1):
        img = flip_img(img)
    img = cv2.resize(img,(160,160))
    return  img


path_to_augmentation = "Data_1"
number_img_augmentation = 300

path_to_augmentation1 = "Data_1_output"

folders = os.listdir(path_to_augmentation)
for folder in folders:
    path_parent = path_to_augmentation + "/" + folder
    path_parent1 = path_to_augmentation1 + "/" + folder
    if not(os.path.isdir(path_parent1)):
        os.mkdir(path_parent1)
    files = os.listdir(path_parent)
    for file in files:
        path = path_parent +"/"+ file
        path1 = path_parent1 +"/"+ file
        img = cv2.imread(path)
        for i in range(number_img_augmentation):
            img_augmentation = augmentation_img(img)
            path_save  = path1[:-4] + str(i) + ".jpg"
            cv2.imwrite(path_save,img_augmentation)




        # cv2.imshow("img", img)
        # break
    # break

# cv2.waitKey(0)
# path = "10detected/" + folders[0]
# files = os.listdir(path)

# # for file in folders:


# print(list(files))




# img = cv2.imread("192.168.1.47_01_20220719123716617_FACE_SNAP.png")
# for i in range(1):
#     img_augmentation = augmentation_img(img)
#     print(img_augmentation.shape)
#     cv2.imshow(f"fff{i}",img_augmentation)






# cv2.imshow("img1",img)
# img = augmentation_img(img)
# cv2.imshow("fdf", img)


# print(img)
# cv2.waitKey(0)


# print(random_value())