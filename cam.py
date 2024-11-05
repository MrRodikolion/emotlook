from multiprocessing import Process, Value, Array, Queue
import ctypes

import torch
from torch import load, cuda, no_grad, argmax
from torch.nn import Linear
from torch.nn.functional import softmax
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import cv2
import numpy as np

from time import time


class CamProcess(Process):
    def __init__(self):
        super().__init__()
        self.shape = (480, 640, 3)

        h, w, c = self.shape
        self.frame = Array(ctypes.c_uint8, h * w * c)
        self.netframe = Array(ctypes.c_uint8, h * w * c)

        self.cur_emotes = Array('i', [-1] * 100)

        self.in_que = Queue()
        self.out_que = Queue()

        self.started = False

    def run(self):
        super().run()
        class_name = ['angry', 'confused', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'shy', 'surprise']

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        _, frame = cap.read()
        self.out_que.put(frame)

        emotes_list = []

        faces = []

        self.started = True
        while True:
            _, frame = cap.read()
            draw_frame = frame.copy()

            if not self.in_que.empty():
                faces = self.in_que.get()
                self.out_que.put(frame)

            for face in faces:
                try:
                    pt1 = face[0]
                    pt2 = face[1]
                    emote_id = face[2]
                    cv2.rectangle(draw_frame, pt1, pt2, (0, 0, 255), 2)
                    cv2.putText(draw_frame, class_name[emote_id], pt1, cv2.FONT_ITALIC, 2, (0, 0, 255), 5)

                    emotes_list.append(emote_id)
                    if len(emotes_list) > 100:
                        del emotes_list[0]
                except BaseException as e:
                    print(e)

            for i, emote in enumerate(emotes_list):
                self.cur_emotes[i] = emote
            np.copyto(np.frombuffer(self.frame.get_obj(), dtype=np.uint8).reshape(frame.shape), frame)
            np.copyto(np.frombuffer(self.netframe.get_obj(), dtype=np.uint8).reshape(frame.shape), draw_frame)
