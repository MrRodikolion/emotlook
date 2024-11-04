from multiprocessing import Process, Value, Array
import ctypes

from flask import Flask, render_template, Response, send_file, redirect, jsonify
import cv2
from PIL import Image
import numpy as np
import base64
import io

from collections import Counter, OrderedDict

from cam import CamProcess


class ServerProcess(Process):
    def __init__(self, camth: CamProcess):
        super().__init__()
        self.camth = camth

    def run(self):
        super().run()
        class_name = ['angry', 'confused', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'shy', 'surprise']

        app = Flask(__name__)

        @app.route('/index_old')
        def index_old():
            return render_template('index_old.html')

        @app.route('/getframe_old')
        def get_frame_old():
            def generate():
                while True:
                    frame = np.frombuffer(self.camth.frame.get_obj(), dtype=np.uint8).reshape(self.camth.shape)
                    if frame is None:
                        continue
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/getnetframe_old')
        def get_netframe_old():
            def generate():
                while True:
                    frame = np.frombuffer(self.camth.netframe.get_obj(), dtype=np.uint8).reshape(self.camth.shape)
                    if frame is None:
                        continue
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/get_frame')
        def get_frame():
            frame = np.frombuffer(self.camth.frame.get_obj(), dtype=np.uint8).reshape(self.camth.shape)
            h, w, c = frame.shape
            frame = cv2.resize(frame, (w // 2, h // 2))

            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode()

            return jsonify({'image': img_str})

        @app.route('/get_netframe')
        def get_netframe():
            frame = np.frombuffer(self.camth.netframe.get_obj(), dtype=np.uint8).reshape(self.camth.shape)
            h, w, c = frame.shape
            frame = cv2.resize(frame, (w // 2, h // 2))

            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode()

            return jsonify({'image': img_str})

        @app.route('/get_emotes')
        def get_emotes():
            emotes = [i for i in self.camth.cur_emotes if i != -1]
            counts = OrderedDict(sorted(Counter(emotes).items()))
            names = [class_name[k] for k in counts.keys()]
            data = list(counts.values())

            return jsonify({
                'names': names,
                'data': data
            })

        app.run(host='0.0.0.0', port=5000)
