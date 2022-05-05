# Run this script so the program would start and then go to the http://127.0.0.1:5000 site and see your results :))

from flask import Flask, render_template, Response, jsonify
from FaceRecognition import showcamera
import cv2

app = Flask(__name__)

video_stream = showcamera() # Iš failo VeidoAtpazinimas.py pašaukiama klasė showcamera

@app.route('/')
def index():
    return render_template("index.html")

def gen(VaizdoAtpazinimas):
    while True: # Jeigu programa veikia gaunami kadrai
        frame = VaizdoAtpazinimas.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
@app.route('/video_feed')

def video_feed():
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True,port="5000") # sukuriamas lokalus tinklapis
