from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ast import While, arg
from dataclasses import dataclass
from tkinter.messagebox import QUESTION
from tkinter.tix import Tree
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
import time
import argparse
import facenet
import imutils
import os
import sys
import math
import detecter
from PIL import ImageFont, ImageDraw, Image
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import time
import threading
# from retinafacemaster.retinaface import RetinaFace 
import math
import numpy as np 
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
import keyboard

global queus1
queus1 = set()
global queus2
queus2 = set()

# remove accent when saving

def remove_accent(text):
    return unidecode.unidecode(text)


def save_img_checkin(img,name_staff):
    path_img = f"{name_staff}.jpg"
    path_img_checkin = "images/checkin/" + str(datetime.date.today())
    if not(os.path.exists(path_img_checkin)):
        os.mkdir(path_img_checkin)
    file_img_name_path = path_img_checkin +"/" + path_img
    cv2.imwrite(str(file_img_name_path),img)
    return file_img_name_path


def save_img_checkout(img,name_staff):
    path_img = f"{name_staff}.jpg"
    path_img_checkin = "images/checkout/" + str(datetime.date.today())
    if not(os.path.exists(path_img_checkin)):
        os.mkdir(path_img_checkin)
    file_img_name_path = path_img_checkin +"/" + path_img
    cv2.imwrite(str(file_img_name_path),img)
    return file_img_name_path


def has_existed(database,ma_nhan_vien):
    query = f"SELECT id from checkin_checkout WHERE date(datetime) == date('now','localtime') and ma_nhan_vien == '{ma_nhan_vien}'"
    a = database.execute(query)
    database.commit()
    return list(a)


def save_img_unknown(img,ma_nhan_vien):
    date_time = str(datetime.datetime.now().time()).replace(":","")
    path_img = f"{ma_nhan_vien}{date_time}.jpg"
    path_img_checkin = "images/unknown/" + str(datetime.date.today())
    if not(os.path.exists(path_img_checkin)):
        os.mkdir(path_img_checkin)
    file_img_name_path = path_img_checkin +"/" + path_img
    cv2.imwrite(str(file_img_name_path),img)
    return 0


# function insert info to database
def insert_infor(img,ma_nhan_vien,ten_nhan_vien, accuracy):
    global queus1
    global queus2
    path = "database/data_base.sql"
    database = sqlite3.connect(path)
    infor_staff_in_database = has_existed(database, ma_nhan_vien)
    if len(infor_staff_in_database) == 0:
        queus1.add(" Xin chào " + str(ten_nhan_vien) + " ")
        queus2.add(str(ten_nhan_vien))
        path_img = save_img_checkin(img,remove_accent(ten_nhan_vien))
        query  = f"""
                        INSERT INTO checkin_checkout(ma_nhan_vien,accuracy,datetime,image_path) 
                        VALUES('{ma_nhan_vien}','{accuracy}',datetime('now', 'localtime'),'{path_img}')
                    """
        database.execute(query)
        database.commit()
        database.close()
    elif len(infor_staff_in_database) == 1: 
        queus1.add(" Tạm biệt " + str(ten_nhan_vien) + " ")
        queus2.add(str(ten_nhan_vien))
        path_img = save_img_checkout(img,remove_accent(ten_nhan_vien))
        query  = f"""
                        INSERT INTO checkin_checkout(ma_nhan_vien,accuracy,datetime,image_path) 
                        VALUES('{ma_nhan_vien}','{accuracy}',datetime('now', 'localtime'),'{path_img}')
                    """
        database.execute(query)
        database.commit()
        database.close()
    else:
        queus1.add(" Tạm biệt " + str(ten_nhan_vien) + " ")
        queus2.add(str(ten_nhan_vien))
        
        # print("checkin_checkout: ",queus)
        path_img = save_img_checkout(img,remove_accent(ten_nhan_vien))
        query  = f"""
                        UPDATE checkin_checkout 
                        SET datetime = datetime('now','localtime'),
                        accuracy = '{accuracy}', image_path = '{path_img}' 
                        WHERE id = {infor_staff_in_database[1][0]}
                    """
        database.execute(query)
        database.commit()
        database.close()
    return 0


# take information of staff in company from database
def take_info_database(ma_nhan_vien):
    path = "database/data_base.sql"
    database = sqlite3.connect(path)
    query = f"SELECT ho_va_ten,vi_tri FROM information_staff WHERE ma_nhan_vien=='{ma_nhan_vien}'"
    infor = database.execute(query)
    infor = list(infor)[0]
    database.commit()
    database.close()
    return infor


class checkin_checkout(threading.Thread):
    def __init__(self, src, precision, *args, **kwargs):

        super(checkin_checkout, self).__init__(*args, **kwargs)
        self._stopper = threading.Event()
    # def __init__(self, src,precision):
        self.CLASSIFIER_PATH = 'Models/facemodel.pkl'
        self.FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
        self.INPUT_IMAGE_SIZE = 160
        self.src = src
        self.precision = precision
    
    def stop(self):
        self._stopper.set()

    def stopped(self):
        return self._stopper.isSet()

    def run(self):
        global queus1

        with open(self.CLASSIFIER_PATH, 'rb') as file:
            model, class_names = pickle.load(file)
        print("Custom Classifier, Successfully loaded")

        with tf.Graph().as_default():
            gpu_options = tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=0.6)
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                gpu_options=gpu_options, log_device_placement=False))

            with sess.as_default(): 

                detection = detecter.human_mask()
                facenet.load_model(self.FACENET_MODEL_PATH)
                # Get input and output tensors
                images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

                cap = VideoStream(self.src).start()

                while (True):
                    t1 = time.time()
                    try:
                        frame = cap.read()
                        frame = imutils.resize(frame, width=960*2)
                        frame = imutils.resize(frame, height=540*2)
                        (locs, preds) = detection.detect_and_predict_mask(frame)

                       

                        bb = np.zeros((1, 4), dtype=np.int32)
                        i =0

                        for (box,pred) in zip(locs, preds):
                            # print(det[i][2] - det[i][0])
                            # print(det[i][3] - det[i][1])
                            # if ((det[i][2] - det[i][0] < 10) or (det[i][3] - det[i][1] < 20)) or ((det[i][2] - det[i][0] > 150) or (det[i][3] - det[i][1] > 160)) :
                            #     continue
                            (mask, withoutMask) = pred

                            if mask > withoutMask:
                                print('masked')
                                continue

                            (startX, startY, endX, endY) = box
                            bb[i][0] = startX
                            bb[i][1] = startY
                            bb[i][2] = endX
                            bb[i][3] = endY         
                            cropped = frame[bb[i][1]:bb[i]
                                            [3], bb[i][0]:bb[i][2], :]
                            
                            scaled = cv2.resize(cropped, (self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(
                                -1, self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 3)
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

                            print("Name: {}, Probability: {}".format(
                                best_name, best_class_probabilities))

                            if best_class_probabilities > self.precision:
                                cv2.rectangle(
                                    frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                ma_nhan_vien = class_names[best_class_indices[0]]
                                infor = take_info_database(ma_nhan_vien)
                                insert_infor(frame[bb[i][1]-50:bb[i]
                                            [3]+50, bb[i][0]-50:bb[i][2]+50, :],ma_nhan_vien ,infor[0], best_class_probabilities[0])

    # ---------------------------------------------------------------------------------------------------------------------------------

                                info_clerk = infor[0] + '\n'+ infor[1]
                                fontpath = r"src\arial.ttf"   
                                font = ImageFont.truetype(fontpath, 15)
                                img_pil = Image.fromarray(frame)
                                draw = ImageDraw.Draw(img_pil)
                                draw.text((text_x, text_y),  (info_clerk), font = font, fill = (255, 215, 0))
                                frame = np.array(img_pil)

    # --------------------------------------------------------------------------------------------------------------------------------    
    #                          
                            elif best_class_probabilities > 0.4:
                                save_img_unknown(frame[bb[i][1]:bb[i]
                                            [3], bb[i][0]:bb[i][2]] , class_names[best_class_indices[0]])

    # --------------------------------------------------------------------------------------------------------------------------------

                                cv2.rectangle(
                                    frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                cv2.putText(frame, "Unknown Person", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 255), thickness=1, lineType=2)

    # ---------------------------------------------------------------------------------------------------------------------------------

                    

                        cv2.imshow('camera', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except: 
                        cap = VideoStream(self.src).start()
                    print(f"FPS: {1/(time.time() - t1) }")

                cap.stop()
                cv2.destroyAllWindows()


class alert(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(alert, self).__init__(*args, **kwargs)
    #     self._stopper = threading.Event()
        
    # def stop(self):
    #     self._stopper.set()
 
    # def stopped(self):
    #     return self._stopper.isSet()

    def run(self):
        global queus1
        global queus2
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        vi_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_viVN_An"
        engine.setProperty("voice", vi_voice_id)
        engine.setProperty("rate", 180)
        time_delay = 0
        saving = []
        
        while True:
            # time_delay1 = time.time()
            # lst =list()

            # while (time.time() - time_delay1 ) < 10 :
            # while True:
            continue


            try:
                # i = []
                # lst = list(queus) + lst

                # if keyboard.is_pressed("q"):
                #     self.stop()
                # if self.stopped():
                #     return 0

                continue
                
                for item in queus1:
                    # saving.append(item)
                    # if item in lst:
                    #     queus.remove(item)
                    # print
                    print(item)
                        # continue

                    # if datetime.datetime.now().hour >= 12:          
                    #     i.append(" Tạm biệt " + item + " ")
                    # else: 
                    #     i.append(" Xin chào " + item + " ")
                    # engine.say((i.pop(0)))
                    # queus.remove(item)
                    # engine.runAndWait()

            
                

            except Exception as e:
                print(e)
                queus1.add("")   


    

al = checkin_checkout(src = 0, precision= 0.8)
# src=( r"rtsp://admin:sp@ce123@192.168.1.49:554")
# ths = alert()

al.start()
# ths.start()










