from itertools import count
from multiprocessing.sharedctypes import Value
from unittest import result
import cv2
import pickle


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Duomenys veido atpazinimui is Haar Cascade github
recognizer = cv2.face.LBPHFaceRecognizer_create() # Sukuriamas veido atpažinmo kintamasis
recognizer.read("trainner.yml") # Nuskaitomas veido atpazinimo failas sugeneruotas iš veido atpažįnimo mokymų

labels = {"person_name": 1} # Sukuriama etiketė vardams
with open("labels.pickle","rb") as f:
    og_labels = pickle.load(f) # Nuskaitomos visos etiketės
    labels = {v:k for k,v in og_labels.items()} #inverting the labels

def detectandshow(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Pakeičiama į pilkiamatį vaizdą, kad lengviau būtų atpažinti veidą
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        id_, conf = recognizer.predict(roi_gray) # Gaunamas atpažinimo sistemos pasitikėjimas
        if conf>=55 and conf <= 85:  # Jeigu atpazinimo sistemos pasitikėjimas viršyja 45 ir nera daugiau 85
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color= (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x, y),font, 1, color, stroke, cv2.LINE_AA) # Priedama vardo etiketė virš nustatytų veidų
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Sukuriamas žalias stačiakampis aplinkui veidą
        elif conf>=0:  # Jeigu atpazinimo sistemos pasitikėjimas viršyja 0 ir nera daugiau 45
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # Sukuriamas raudonas stačiakampis aplinkui veidą
    cv2.waitKey(10)  # Laukiama, kad neužkrauti sistemos

class showcamera: # Kameros klasė

    def __init__(self):
        self.video = cv2.VideoCapture(0) # Pradeda vaizdo nuskaitymą

    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret, frame = self.video.read() # Frame kintamasis nuskaito kadrą
        detectandshow(frame)
        ret,jpeg = cv2.imencode('.jpg',frame) # Išvedamas kadras .jpg formatu, kad būtų lengviau perkelti į lokalų tinklapį
           
        return jpeg.tobytes()

