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

        self.doSingleImg = Value('b', False)
        self.img = Array(ctypes.c_uint8, h * w * c)

        self.started = False

    def run(self):
        super().run()
        class_name = ['angry', 'confused', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'shy', 'surprise']

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        net_proc = NetProcess()
        net_proc.start()

        self.started = True
        while True:
            _, frame = cap.read()
            np.copyto(np.frombuffer(net_proc.frame.get_obj(), dtype=np.uint8).reshape(frame.shape), frame)

            draw_frame = frame.copy()
            if net_proc.isDone.value:
                try:
                    pt1 = tuple(int(c) for c in net_proc.pt1.get_obj())
                    pt2 = tuple(int(c) for c in net_proc.pt2.get_obj())
                    emote_id = int(net_proc.emote_id.value)
                    cv2.rectangle(draw_frame, pt1, pt2, (0, 0, 255), 2)
                    cv2.putText(draw_frame, class_name[emote_id], pt1, cv2.FONT_ITALIC, 2, (0, 0, 255), 5)
                except BaseException as e:
                    print(e)

            for i, emote in enumerate(net_proc.cur_emotes.get_obj()):
                self.cur_emotes[i] = emote

            np.copyto(np.frombuffer(self.frame.get_obj(), dtype=np.uint8).reshape(frame.shape), frame)
            np.copyto(np.frombuffer(self.netframe.get_obj(), dtype=np.uint8).reshape(frame.shape), draw_frame)


class NetProcess(Process):
    def __init__(self):
        super().__init__()
        self.shape = (480, 640, 3)

        h, w, c = self.shape
        self.frame = Array(ctypes.c_uint8, h * w * c)
        self.out = Queue()

        self.cur_emotes = Array('i', [-1] * 100)

        self.pt1 = Array('i', [0, 0])
        self.pt2 = Array('i', [0, 0])
        self.emote_id = Value('i', -1)
        self.isDone = Value('b', False)

        self.isNewFrame = False
        self.started = False

    def run(self):
        super().run()

        device = "cuda" if cuda.is_available() else "cpu"

        model = models.resnet50()

        class_name = ['angry', 'confused', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'shy', 'surprise']

        emotes_list = []

        num_class = len(class_name)
        model.fc = Linear(model.fc.in_features, num_class)

        model.to(device)
        model.load_state_dict(load(r"./emogion_last.pth", map_location=device))
        model.eval()
        preporcec = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        faceNet = cv2.dnn.readNet("./opencv_face_detector_uint8.pb", "./opencv_face_detector.pbtxt")
        faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.started = True
        while True:
            if self.doSingleImg.value:
                frame = np.frombuffer(self.img.get_obj(), dtype=np.uint8).reshape(self.shape)
                draw_img = frame.copy()
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
                faceNet.setInput(blob)
                detections = faceNet.forward()

                h, w, c = frame.shape
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.7:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)

                        fw = x2 - x1
                        fh = y2 - y1

                        x, y = fw // 2, fh // 2

                        s = max(fw, fh) // 2
                        shift = s // 5

                        face_img = frame[y - s + y1:y + s + y1, x - s + x1:x + s + x1]

                        cv2.rectangle(draw_img, (x - s + x1, y - s + y1), (x + s + x1, y + s + y1), (0, 0, 255), 2)

                        image = Image.fromarray(face_img)
                        image = preporcec(image)
                        image = image.unsqueeze(0).to(device)

                        with no_grad():
                            output: torch.Tensor = model(image)

                        # probabilities = softmax(output[0], dim=0)
                        emote_id = argmax(output[0])
                        emote_id = int(emote_id)
                np.copyto(np.frombuffer(self.frame.get_obj(), dtype=np.uint8).reshape(frame.shape), frame)

            frame = np.frombuffer(self.frame.get_obj(), dtype=np.uint8).reshape(self.shape)
            face_deteced = False
            try:
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
                faceNet.setInput(blob)
                detections = faceNet.forward()

                h, w, c = frame.shape
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.7:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)

                        fw = x2 - x1
                        fh = y2 - y1

                        x, y = fw // 2, fh // 2

                        s = max(fw, fh) // 2
                        shift = s // 5

                        face_img = frame[y - s + y1:y + s + y1, x - s + x1:x + s + x1]

                        # cv2.rectangle(draw_frame, (x - s + x1, y - s + y1), (x + s + x1, y + s + y1), (0, 0, 255), 2)

                        image = Image.fromarray(face_img)
                        image = preporcec(image)
                        image = image.unsqueeze(0).to(device)

                        with no_grad():
                            output: torch.Tensor = model(image)

                        # probabilities = softmax(output[0], dim=0)
                        emote_id = argmax(output[0])
                        emote_id = int(emote_id)

                        face_deteced = True
                        self.pt1[0], self.pt1[1] = (x - s + x1, y - s + y1)
                        self.pt2[0], self.pt2[1] = (x + s + x1, y + s + y1)
                        self.emote_id.value = emote_id

                        emotes_list.append(emote_id)
                        if len(emotes_list) > 100:
                            del emotes_list[0]
                        for n, emote in enumerate(emotes_list):
                            self.cur_emotes[n] = emote
                        break
            except:
                pass

            self.isDone.value = face_deteced
