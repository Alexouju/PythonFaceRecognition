# This is the face recognition training
# Run this file so you could train a program
# Upload pictures to "output_path"
# Then run and from your photos it will understand a persons face


import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gaunama nuoroda failo dabartinei vietai
image_dir = os.path.join(BASE_DIR,"output_path") # Pasirenkamas aplankalas "output_path"

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Duomenys veido atpazinimui is Haar Cascade github
recognizer = cv2.face.LBPHFaceRecognizer_create() # Sukuriamas veido atpažinmo kintamasis


x_train = [] # Ekrano reikšmės veido atpažinimui
y_labels = [] # Galimos etiketės

current_id = 0
label_ids ={}

for root,dirs,files in os.walk(image_dir): 
    for file in files: # Nuskaitomas kiekvienas failas
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if not label in label_ids: # Jei nėra eitiketės label masyve pridedama ir etiketės skaičius padidinamas
                label_ids[label] = current_id
                current_id+=1
            
            id_ =label_ids[label] # Priskiriamas kintamasis etiketėms
            pil_image = Image.open(path).convert("L") # Pakeičiama į pilką toną
            image_array = np.array(pil_image,"uint8")
            faces = faceCascade.detectMultiScale(image_array)

            for (x,y,w,h) in faces: # Kiekvienam veidui pridedamos ekrano reikšmės ir jos etiketės numeris
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


with open("labels.pickle","wb") as f: # Irašomos etiketės
    pickle.dump(label_ids,f) 

recognizer.train(x_train,np.array(y_labels)) # Mokoma
recognizer.save("trainner.yml") # Mokymo duomenys irašomi
