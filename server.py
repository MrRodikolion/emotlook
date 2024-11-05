from multiprocessing import Process, Value, Array
import ctypes

from flask import Flask, render_template, Response, send_file, redirect, jsonify, request
import logging
import cv2
from PIL import Image
import numpy as np
import base64
import io
import os

from simgle_img_net import get_neuroned

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
        SECRET_KEY = os.urandom(32)
        app.config['SECRET_KEY'] = SECRET_KEY
        app.debug = False

        # app.logger.disabled = True
        # log = logging.getLogger('werkzeug')
        # log.disabled = True

        @app.route('/index_old', methods=['GET', 'POST'])
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

        @app.route('/', methods=['GET', 'POST'])
        def index():
            if request.method == 'POST':
                if 'file1' not in request.files:
                    return 'there is no file1 in form!'
                file1 = request.files['file1']
                img = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), -1)

                net_frame = get_neuroned(img)
                h, w, c = net_frame.shape
                net_frame = cv2.resize(net_frame, (w // 2, h // 2))

                _, buffer = cv2.imencode('.jpg', net_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                img_str = base64.b64encode(buffer).decode()

                return jsonify({'image': img_str})

            return render_template('index.html')

        @app.route('/get_frame')
        def get_frame():
            frame = np.frombuffer(self.camth.frame.get_obj(), dtype=np.uint8).reshape(self.camth.shape)
            h, w, c = frame.shape
            frame = cv2.resize(frame, (w // 2, h // 2))

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            img_str = base64.b64encode(buffer).decode()

            return jsonify({'image': img_str})

        @app.route('/get_netframe')
        def get_netframe():
            frame = np.frombuffer(self.camth.netframe.get_obj(), dtype=np.uint8).reshape(self.camth.shape)
            h, w, c = frame.shape
            frame = cv2.resize(frame, (w // 2, h // 2))

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            img_str = base64.b64encode(buffer).decode()

            return jsonify({'image': img_str})

        @app.route('/get_emotes')
        def get_emotes():
            emotes = [i for i in self.camth.cur_emotes.get_obj() if i != -1]
            counts = OrderedDict(sorted(Counter(emotes).items()))
            names = [class_name[k] for k in counts.keys()]
            data = list(counts.values())

            return jsonify({
                'names': names,
                'data': data
            })

        app.run(host='0.0.0.0', port=5000)
