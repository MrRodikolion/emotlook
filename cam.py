from multiprocessing import Process, Value, Array
import ctypes

from torch import load, cuda, no_grad, argmax
from torch.nn import Linear
from torch.nn.functional import softmax
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import cv2
import numpy as np


class CamProcess(Process):
    def __init__(self):
        super().__init__()
        self.shape = (480, 640, 3)

        h, w, c = self.shape
        self.frame = Array(ctypes.c_uint8, h * w * c)
        self.netframe = Array(ctypes.c_uint8, h * w * c)

        self.cur_emotes = Array('i', [-1] * 100)

        self.started = False

    def run(self):
        super().run()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        device = "cuda" if cuda.is_available() else "cpu"

        model = models.resnet50()

        class_name = ['angry', 'confused', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'shy', 'surprise']
        # class_name = ["anger", "contempt", "disgust", "fear", "happiness", "neutrality", "sadness", "surprise"]

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
            _, frame = cap.read()

            draw_frame = frame.copy()
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
                        # face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        # face_img = cv2.merge([face_img, face_img, face_img])

                        cv2.rectangle(draw_frame, (x - s + x1, y - s + y1), (x + s + x1, y + s + y1), (0, 0, 255), 2)

                        image = Image.fromarray(face_img)
                        image = preporcec(image)
                        image = image.unsqueeze(0).to(device)

                        with no_grad():
                            output = model(image)

                        probabilities = softmax(output[0], dim=0).cpu()
                        emote_id = argmax(probabilities)

                        cv2.putText(draw_frame, class_name[emote_id], (x1, y1), cv2.FONT_ITALIC, 2, (0, 0, 255), 5)

                        emotes_list.append(emote_id)
                        if len(emotes_list) > 100:
                            del emotes_list[0]
                        for n, emote in enumerate(emotes_list):
                            self.cur_emotes[n] = emote
            except:
                pass

            np.copyto(np.frombuffer(self.frame.get_obj(), dtype=np.uint8).reshape(frame.shape), frame)
            np.copyto(np.frombuffer(self.netframe.get_obj(), dtype=np.uint8).reshape(frame.shape), draw_frame)
