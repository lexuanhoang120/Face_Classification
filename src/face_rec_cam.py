from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataclasses import dataclass
from tkinter.messagebox import QUESTION
from traceback import print_tb
from cv2 import FONT_HERSHEY_PLAIN, exp
import unidecode
import pyttsx3
import tensorflow as tf
from imutils.video import VideoStream
import datetime
import cv2
import numpy as np
import sqlite3
import os.path
import argparse
import facenet
import imutils
import os
import sys
import math
from PIL import ImageFont, ImageDraw, Image
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
import random
from sklearn.svm import SVC
import time
import threading
# from retinafacemaster.retinaface import RetinaFace 
import math
import numpy as np 
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Python program to
# demonstrate queue implementation
# using collections.dequeue
  
  
from collections import deque

def remove_accent(text):
    return unidecode.unidecode(text)

# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# vi_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_viVN_An"
# engine.setProperty("voice", vi_voice_id)
# engine.setProperty("rate", 178)

    
global queus
# queus = deque()
queus = []


def save_img_checkin(img,ma_nhan_vien):
    # path_img = str(datetime.datetime.now()) + '.jpg'
    path_img = f"{ma_nhan_vien}.jpg"
    path_img_checkin = "checkin/" + str(datetime.date.today())
    if not(os.path.exists(path_img_checkin)):
        os.mkdir(path_img_checkin)
    file_img_name_path = path_img_checkin +"/" + path_img
    # print(file_img_name_path)

    cv2.imwrite(str(file_img_name_path),img)
    return file_img_name_path
# a function 
# function connnect the database
# a function  check that member which exist in checkin

def has_existed(database,ma_nhan_vien):
    query = f"SELECT * from checkin WHERE date(checkin.datetime) == date('now','localtime') and checkin.ma_nhan_vien == '{ma_nhan_vien}'"
    a = database.execute(query)
    database.commit()
    if list(a)==[]:
        return 0
    else:
        return 1
    # print(list(a)==[])

# function insert info to database
def insert_checkin(img,ma_nhan_vien,ten_nhan_vien, accuracy):
    global queus
    # queus = [""]
    # path_img = save_img_checkin(img,ma_nhan_vien)
    path = "database//check_in.sql"
    database = sqlite3.connect(path)
    if has_existed(database,ma_nhan_vien):
        database.close()
    else: 

        inf = " xin chào " + str(ten_nhan_vien) + " "
        queus.append(inf)
        # if len(queus)>1:
        #     queus.drop(0)

        path_img = save_img_checkin(img,remove_accent(ten_nhan_vien))
        query  = f"INSERT INTO checkin (ma_nhan_vien,accuracy,datetime,image_path) VALUES('{ma_nhan_vien}','{accuracy}',datetime('now', 'localtime'),'{path_img}')"
        database.execute(query)
        database.commit()
        database.close()
        
    return 0



#take information of staff in company from database
def take_info_database(ma_nhan_vien):
    path = "database//Data_Employee_VTcode.db"
    database = sqlite3.connect(path)
    query = f"SELECT ho_va_ten,vi_tri FROM Employee WHERE ma_nhan_vien=='{ma_nhan_vien}'"
    infor = database.execute(query)
    infor = list(infor)[0]
    database.commit()
    database.close()
    return infor

# def rotate(image, angle):
#   image_center = tuple(np.array(image.shape[1::-1]) / 2)
#   rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#   result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#   return result

# def rotate_image(image):
#     print("beginfdsfdffff")
#     ree = RetinaFace.detect_faces( image)
#     print("adasdassad")
#     x1, y1 = ree["face_1"]["landmarks"]["right_eye"]
#     x2, y2 = ree["face_1"]["landmarks"]["left_eye"]
#     d = 0
#     if (y1- y2 > 0):
#         d = 1
#     else:
#         d = -1
#     a = abs (y1 - y2)
#     b = abs (x2 - x1)
#     c = math.sqrt(a*a + b*b)
#     cos_alpha = (b*b + c*c - a*a) / (2 * b * c)
#     alpha = np.arccos(cos_alpha)
#     alpha =(alpha*180) / math.pi
#     # aligned_img = Image.fromarray(img)
#     # aligned_img = np.array(aligned_img.rotate( - d * alpha))
#     print(alpha)
#     img  = rotate(image, round(- d * alpha))
#     return img
# global num_img
# num_img = 0

def main():
    global queus
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        # sess = tf.device("/cpu:0")

        with sess.as_default(): 

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph(
            ).get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, model_path="src/align")

            # people_detected = set()
            # person_detected = collections.Counter()
            # cap = cv2.imread('D:\computer_vision\DataSet\FaceData\processed\PhuocNam\Screenshot 2022-07-08 105839.png')
            # cap = VideoStream(src=(
            #     r"rtsp://admin:space123@192.168.1.47:554/cam/realmonitor?channel=1&subtype=1")).start()
            # cap = cv2.VideoCapture(0)
            cap = VideoStream(src=(0)).start()
            while (True):
                frame = cap.read()
                
                # global name
                # global num_img
                # name = '' 
                # x = 2
                # frame = imutils.resize(frame, width=960*2)
                # frame = imutils.resize(frame, height=540*2)
                # print(frame.shape)
                # frame = frame[150*2:,320*2:620*2]
                # print(type(frame))
                # frame = cv2.flip(frame, 1)
                # Cropping an image
                # cropped_image = frame[:, 350:1000]
                # cv2.rectangle(frame,(350,0),(1000,768),(0,255,0),2)


                bounding_boxes, _ = align.detect_face.detect_face(
                    frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                # print(f"Number of faces: {faces_found}")
                try:
                    # if faces_found > 15:
                    #     cv2.putText(frame, "Face more", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    #                 1, (255, 0, 0), thickness=1, lineType=2)
                    if faces_found > 0:
                        
                        
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            # print(det[i][2] - det[i][0])
                            # print(det[i][3] - det[i][1])
                            # if ((det[i][2] - det[i][0] < 10) or (det[i][3] - det[i][1] < 20)) or ((det[i][2] - det[i][0] > 150) or (det[i][3] - det[i][1] > 160)) :
                            #     continue
                            bb[i][0] = det[i][0] 
                            bb[i][1] = det[i][1] 
                            bb[i][2] = det[i][2] 
                            bb[i][3] = det[i][3] 
                            # print(f"{bb[i][1] - bb[i][3]}, {bb[i][0] - bb[i][2]}")
                            # print(bb[i][3]-bb[i][1])
                            # print(frame.shape[0])
                            # print((bb[i][3]-bb[i][1])/frame.shape[0])
                             
                            # print(bb[i] [3] - bb[i][1])
                            # print(frame.shape[0])
                            # print((bb[i][3] - bb[i][1])/frame.shape[0])
                        
                            cropped = frame[bb[i][1]-5:bb[i]
                                            [3]+5, bb[i][0]-5:bb[i][2]+5, :]
                            
                            # print(type(cropped))
                            # cropped = rotate_image(cropped)
                            # print(type(cropped))
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(
                                -1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {
                                images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(
                                embeddings, feed_dict=feed_dict)

                            predictions = model.predict_proba(emb_array)

                            best_class_indices = np.argmax(
                                predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]
                            best_name = class_names[best_class_indices[0]]
                            # name = class_names[best_class_indices[0]]
                            print("Name: {}, Probability: {}".format(
                                best_name, best_class_probabilities))

                            if best_class_probabilities > 0.75:
                                # file_name = "dat2//dat" + str(num_img) + ".jpg"
                                # cv2.imwrite(file_name, cropped)
                                # num_img = num_img + 1
                                cv2.rectangle(
                                    frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                ma_nhan_vien = class_names[best_class_indices[0]]


                                # ma_nhan_vien = name
                                # print(type(ma_nhan_vien))


                                # print(ma_nhan_vien)
                                # ma_nhan_vien = "TTS001"
                                # path_img = str(datetime.datetime.now()) + '.jpg'
                                # path_img_checkin = "checkin/" + str(datetime.date.today())
                                # if not(os.path.exists(path_img_checkin)):
                                #     os.mkdir(path_img_checkin)
                                # file_img_name_path = path_img_checkin +"/" + path_img
                                # print(file_img_name_path)
                                # cv2.imwrite(file_img_name_path,cropped)
                                # file_img_name_path = save_img_checkin(,ma_nhan_vien)
                                # file_img_name_path = save_img_checkin(cropped,ma_nhan_vien)
                                infor = take_info_database(ma_nhan_vien)

                                
                                insert_checkin(frame[bb[i][1]-50:bb[i]
                                            [3]+50, bb[i][0]-50:bb[i][2]+50, :],ma_nhan_vien,infor[0], best_class_probabilities[0])

                                

                                info_clerk = infor[0] + '\n'+ infor[1]
                    
                                fontpath = r"D:\comp_vision\main\src\arial.ttf"     
                                font = ImageFont.truetype(fontpath, 15)
                                img_pil = Image.fromarray(frame)
                                draw = ImageDraw.Draw(img_pil)
                                draw.text((text_x, text_y),  (info_clerk), font = font, fill = (255, 215, 0))
                                frame = np.array(img_pil)
                                # cv2.putText(frame, ma_nhan_vien, (text_x, text_y),cv2.FONT_HERSHEY_PLAIN, 1 ,(255, 215, 0), thickness=1, lineType=2)
                                # cv2.putText(frame,, (text_x, text_y + 17),
                                #             cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #             1, (255, 215, 0), thickness=1, lineType=2)
                                # person_detected[best_name] += 1
                                # global name

                                # name = ' Xin chào ' + infor[0]

                                # mythread2.start()
                                # mythread2.join()
                                # engine.say(name)
                                # engine.runAndWait()



                                # engine = pyttsx3.init()
                                # voices = engine.getProperty('voices')
                                # vi_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_viVN_An"
                                # engine.setProperty("voice", vi_voice_id)
                                # engine.setProperty("rate", 178)
                                # engine.say(str(name))
                                # engine.runAndWait()
                                

                            elif best_class_probabilities > 0.4:
                                ma_nhan_vien = str(class_names[best_class_indices[0]])
                                date_time = str(datetime.datetime.now().time()).replace(":","")
                                path_img1 = f"{ma_nhan_vien}{date_time}.jpg"
                                path_img_checkin1 = "images/" + str(datetime.date.today())
                                if not(os.path.exists(path_img_checkin1)):
                                    os.mkdir(path_img_checkin1)
                                file_img_name_path1 = path_img_checkin1 +"/" + path_img1
                                print(file_img_name_path1)

                                cv2.imwrite(str(file_img_name_path1),frame[bb[i][1]-15:bb[i]
                                            [3]+15, bb[i][0]-15:bb[i][2]+15] )
                                # path_save_all = "" + str(datetime.datetime.now()).replace(" ","|")+ ".jpg"


                                # cv2.imwrite(str(path_save_all),)
                                cv2.rectangle(
                                    frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                info_clerk = "Unknown Person"
                                cv2.putText(frame, info_clerk, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 255), thickness=1, lineType=2)
                    else:
                        # name = ''
                        pass
                                
                except: 
                    # name = ''
                    pass

                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            # cap.stream.release()
            cv2.destroyAllWindows()

# main()
# name = ''
# queus.append("")
def alert_checkin():
    global queus
    # queus.append("")
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    vi_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_viVN_An"
    engine.setProperty("voice", vi_voice_id)
    engine.setProperty("rate", 180)

        # queus.append("1")
    while True:
        try:
            # queus.append("1")
            
            
            
            # print(queus)
            # name = queus[0]
            
            # name =queus.popleft()
            # print(queus)
            # print(queus)
            
            engine.say(str(queus.pop(0)))
            # engine.say("1")/
            # queus.pop(0)
            engine.runAndWait()
        except:
            queus.append("")


mythread1 = threading.Thread(target= alert_checkin,name="mythread1")
mythread2 = threading.Thread(target= main, name= 'main')

mythread1.start()
mythread2.start()

# mythread1.join()
# mythread2.join()

# mythread1.stop()
# mythread2.stop()
