from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ast import arg
from dataclasses import dataclass
import multiprocessing
from tkinter.messagebox import QUESTION
from tkinter.tix import Tree
from traceback import print_tb
from unicodedata import name
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
import post_1office
# import multiprocessing

queus = set()

# remove accent when saving


def remove_accent(text):
    return unidecode.unidecode(text)


def save_img_checkin(img, name_staff, code_staff):
    # post_1office.post_checkin_1office(code_staff)
    path_img = f"{name_staff}.jpg"
    path_img_checkin = "images/checkin/" + str(datetime.date.today())
    if not(os.path.exists(path_img_checkin)):
        os.mkdir(path_img_checkin)
    file_img_name_path = path_img_checkin + "/" + path_img
    cv2.imwrite(str(file_img_name_path), img)
    del path_img, path_img_checkin
    return file_img_name_path


def save_img_checkout(img, name_staff, code_staff):
    path_img = f"{name_staff}.jpg"
    path_img_checkout = "images/checkout/" + str(datetime.date.today())
    if not(os.path.exists(path_img_checkout)):
        os.mkdir(path_img_checkout)
    file_img_name_path = path_img_checkout + "/" + path_img
    cv2.imwrite(str(file_img_name_path), img)
    del path_img, path_img_checkout
    return file_img_name_path


def save_img_unknown(img, ma_nhan_vien):
    date_time = str(datetime.datetime.now().time()).replace(":", "")
    path_img = f"{ma_nhan_vien}{date_time}.jpg"
    path_img_unknown = "images/unknown/" + str(datetime.date.today())
    if not(os.path.exists(path_img_unknown)):
        os.mkdir(path_img_unknown)
    file_img_name_path = path_img_unknown + "/" + path_img
    cv2.imwrite(str(file_img_name_path), img)
    del date_time,path_img, path_img_unknown, file_img_name_path
    return 0



def has_existed_checkin(database, ma_nhan_vien):
    a = database.execute(f"SELECT id from checkin WHERE date(datetime) == date('now','localtime') and ma_nhan_vien == '{ma_nhan_vien}'")
    database.commit()
    return list(a)

def has_existed_checkout(database, ma_nhan_vien):
    a = database.execute(f"SELECT id from checkout WHERE date(datetime) == date('now','localtime') and ma_nhan_vien == '{ma_nhan_vien}'")
    database.commit()
    return list(a)





# function insert info to database
def insert_infor(img, ma_nhan_vien, ten_nhan_vien, accuracy):
    global queus
    path = "database/data_base.sql"
    database = sqlite3.connect(path)
    infor_checkout = has_existed_checkout(database, ma_nhan_vien)
    infor_checkin = has_existed_checkin(database, ma_nhan_vien)
    if  len(infor_checkin) == 0:
        post_1office.post_checkin_1office(int(str(ma_nhan_vien)[6:]))
        queus.add(ten_nhan_vien)
        path_img = save_img_checkin(img, remove_accent(
            ten_nhan_vien), int(ma_nhan_vien[6:]))
        query = f"""
                        INSERT INTO checkin(ma_nhan_vien,accuracy,datetime,image_path) 
                        VALUES('{ma_nhan_vien}','{accuracy}',datetime('now', 'localtime'),'{path_img}')
                    """
        database.execute(query)
        database.commit()
        database.close()

    elif len(infor_checkout) == 0:
        queus.add(ten_nhan_vien)
        path_img = save_img_checkout(img, remove_accent(
            ten_nhan_vien), int(ma_nhan_vien[6:]))
        query = f"""
                        INSERT INTO checkout(ma_nhan_vien,accuracy,datetime,image_path) 
                        VALUES('{ma_nhan_vien}','{accuracy}',datetime('now', 'localtime'),'{path_img}')
                    """
        database.execute(query)
        database.commit()
        database.close()

    elif len(infor_checkout) == 1:
        queus.add(ten_nhan_vien)
        path_img = save_img_checkout(img, remove_accent(
            ten_nhan_vien), int(ma_nhan_vien[6:]))
        query = f"""
                        UPDATE checkout 
                        SET datetime = datetime('now','localtime'),
                        accuracy = '{accuracy}', image_path = '{path_img}' 
                        WHERE id = {infor_checkout[0][0]}
                    """
        database.execute(query)
        database.commit()
        database.close()
    del path, database, path_img, query, infor_checkout
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
    del path, database
    return infor



CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
INPUT_IMAGE_SIZE = 160


def run(src, precision):

    global queus
    # queus = set()

    with open(CLASSIFIER_PATH, 'rb') as file:

        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            detection = detecter.human_mask()
            facenet.load_model(FACENET_MODEL_PATH)
            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph(
            ).get_tensor_by_name("phase_train:0")

            cap = VideoStream(src).start()

            while (True):
                try:
                    t1 = time.time()
                    frame = cap.read()
                    frame = imutils.resize(frame, width=960)
                    frame = imutils.resize(frame, height=540)
                    (locs, preds) = detection.detect_and_predict_mask(frame)

                    bb = np.zeros((1, 4), dtype=np.int32)
                    i = 0

                    for (box, pred) in zip(locs, preds):
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

                        print("Name: {}, Probability: {}".format(
                            best_name, best_class_probabilities))

                        if best_class_probabilities > precision:
                            cv2.rectangle(
                                frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            ma_nhan_vien = class_names[best_class_indices[0]]

                            infor = take_info_database(ma_nhan_vien)

                            insert_infor(frame[bb[i][1]-50:bb[i]
                                                [3]+50, bb[i][0]-50:bb[i][2]+50, :], ma_nhan_vien, str(infor[0]), best_class_probabilities[0])

# ---------------------------------------------------------------------------------------------------------------------------------

                            info_clerk = infor[0] + '\n' + infor[1]
                            fontpath = r"src\arial.ttf"
                            font = ImageFont.truetype(fontpath, 15)
                            img_pil = Image.fromarray(frame)
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((text_x, text_y),  (info_clerk),
                                        font=font, fill=(255, 215, 0))
                            frame = np.array(img_pil)

# --------------------------------------------------------------------------------------------------------------------------------
#
                        elif best_class_probabilities > 0.4:
                            save_img_unknown(frame[bb[i][1]:bb[i]
                                                    [3], bb[i][0]:bb[i][2]], class_names[best_class_indices[0]])

# --------------------------------------------------------------------------------------------------------------------------------

                            cv2.rectangle(
                                frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            cv2.putText(frame, "Unknown Person", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)

# ---------------------------------------------------------------------------------------------------------------------------------

                        # cv2.imshow('camera', frame)
                        print(f"FPS: {1/(time.time() - t1) }")
                    # del frame
                    cv2.imshow('camera', frame)
                    

                except Exception as e:
                    print(e)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

            
            cap.stop()
            cv2.destroyAllWindows()




def alert():
    # queus = set()
    engine = pyttsx3.init()
    # voices = engine.getProperty('voices')
    vi_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_viVN_An"
    engine.setProperty("voice", vi_voice_id)
    engine.setProperty("rate", 180)
    while True:
        t1 = time.time()
        saving_name = list()
        time.sleep(0.1)
        
        while ((time.time() - t1) < 30):
            try:
                if keyboard.is_pressed("q"):
                    time.sleep(0.1)
                    return 0              
                if len(queus) == 0:
                    time.sleep(0.1)
                    continue
                else:
                    for que in queus:
                        if que in saving_name:
                            queus.remove(que)
                            time.sleep(0.1)
                            continue
                        else: 
                            if datetime.datetime.now().hour > 12:
                                speech = " Tạm biệt " + str(que)
                            else:
                                speech = " Xin chào " + str(que)
                            queus.remove(que)
                            saving_name.append(que)
                            engine.say(speech)
                            engine.runAndWait()
            except Exception as e:
                # pass
                # queus = set(
                print(e)
                

def post_to_1office():
    while True:
        time.sleep(35)
        if datetime.datetime.now().strftime("%H:%M") == "22:00":
            path = "database/data_base.sql"
            database = sqlite3.connect(path)
            query  = """
            SELECT ma_nhan_vien, datetime 
            FROM checkout 
            WHERE date(datetime) == date('now','localtime')
            """
            informations = database.execute(query)
            database.commit()
            for information in informations:
                post_1office.post_checkout_1office(int(information[0][6:]),information[1])
                time.sleep(1)
            database.close()
            del informations, path, database,query





if __name__ == "__main__":
    
    src = 0
    # src=( r"rtsp://admin:sp@ce123@192.168.1.49:554/cam/realmonitor?channel=1&subtype=1")
    # src=( r"rtsp://admin:sp@ce123@192.168.1.49:554")
    precision = 0.953
    t1 = threading.Thread(target=run, name="1", args= (src, precision))
    t2 = threading.Thread(target=alert,name="2", args=())
    t3 = multiprocessing.Process(target=post_to_1office, name= "3")
    t1.start()
    t2.start()
    t3.start()


 