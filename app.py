from flask import Flask, render_template, Response, send_file
import cv2 as cv2
from torch import load, cuda, no_grad, argmax
from torch.nn import Linear
from torch.nn.functional import softmax
import torchvision.models as models
import torchvision.transforms as transforms
import warnings

from PIL import Image
import numpy as np
import io

warnings.simplefilter(action='ignore', category=FutureWarning)

# initialize the Flask app
app = Flask(__name__)
cap = cv2.VideoCapture(0)
list_1 = []
device = "cuda" if cuda.is_available() else "cpu"

model = models.resnet50()

class_name = ['angry', 'confused', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'shy', 'surprise']
# class_name = ["anger", "contempt", "disgust", "fear", "happiness", "neutrality", "sadness", "surprise"]

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


def neiro(frame):
    draw_frame = frame.copy()
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

    return draw_frame


def gen_frames1():
    while True:
        success, frame = cap.read()
        frame = neiro(frame)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_frames2():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def get_frame():
    return Response(gen_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed1')
def get_netframe():
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
