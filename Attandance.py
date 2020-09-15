import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Images_Attandance'
imageList = []
nameList = []

my_list = os.listdir(path)
print(my_list)

for person in my_list:
    currentImage = cv2.imread(f'{path}/{person}')
    imageList.append(currentImage)
    nameList.append(os.path.splitext(person)[0])


print(nameList)
print(len(imageList))


def findEncodings(imageList):
    encodeListKnow = []
    for item in imageList:
        photograph = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(photograph)[0]
        encodeListKnow.append(encode)
    return encodeListKnow


print("Encoding of all images complete ... ")

def markAttandance(name):
    with open('Attandance.csv', 'r+') as register:
        myDataList = register.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            dateStr = datetime.now().strftime(' %H:%M:%S')
            register.writelines(f'\n{name},{dateStr}')

encodeListKnown = findEncodings(imageList)
#3rd step to find matches in our encodings

capture = cv2.VideoCapture(0)
capture.set(3,300)
capture.set(4,300)

while True:
    #success, img = capture.read()
    #img_small = cv2.resize(img, (0,0), None, 0.5, 0.5)
    success, img_small = capture.read()
    photo = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    currentFrame_FacePos = face_recognition.face_locations(photo)    # this is a 2d array
    currentFrame_Encoding = face_recognition.face_encodings(photo, currentFrame_FacePos)       # this is a list

    for encode_face, face_p in zip(currentFrame_Encoding, currentFrame_FacePos):
        matches = face_recognition.compare_faces(encodeListKnown, encode_face)
        face_dist = face_recognition.face_distance(encodeListKnown, encode_face)
        print(face_dist)
        print(matches)
        print(min(face_dist))
        matchIndex = np.argmin(face_dist)

        y1,x2,y2,x1 = face_p

        if matches[matchIndex] and face_dist[matchIndex] <= 61.0:
            person_name = nameList[matchIndex]
            markAttandance(person_name)
            cv2.rectangle(img_small, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.rectangle(img_small, (x1, y2-35), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(img_small, person_name.upper(), (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
            cv2.imshow("video", img_small)
            cv2.waitKey(35)

        else:
            cv2.rectangle(img_small, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.rectangle(img_small, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(img_small, "---", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255, 0, 0), 2)
            cv2.imshow("video", img_small)
            cv2.waitKey(35)


