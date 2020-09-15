import cv2
import numpy as np
import tensorflow as tf
import face_recognition

imgElon = face_recognition.load_image_file("Resources/Elon.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("Resources/Test.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# finding the faces in our image and finding their encodings as well

faceLocation = face_recognition.face_locations(imgElon)[0]  # sending in a single image... so we are getting first element of this
print(faceLocation)
# encoding the face
encodeElon = face_recognition.face_encodings(imgElon)[0]
print("sdnkjsnvkbdsfkvjbdfkvbskdjf")
print(encodeElon)
# to knw where we have detected the faces...
cv2.rectangle(imgElon,   (faceLocation[3 ],faceLocation[0 ]), (faceLocation[1],faceLocation[1]), (255,0,255), 2)

faceLocationTest = face_recognition.face_locations(imgTest)[0]  # sending in a single image... so we are getting first element of this
print(faceLocationTest)
# encoding the face
encodeTest = face_recognition.face_encodings(imgTest)[0]
# to knw where we have detected the faces...
cv2.rectangle(imgTest,   (faceLocation[3 ],faceLocation[0 ]), (faceLocation[1],faceLocation[1]), (255,0,255), 2)


# third and final step... comparing faces and finding dist bw them ... we have to give in a list of faces

results = face_recognition.compare_faces([encodeElon], encodeTest)
# lower the distance better the match
faceDistance = face_recognition.face_distance([encodeElon], encodeTest )
print(faceDistance)
print(results)

cv2.putText(imgTest, f'{results} {round(faceDistance[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0,112,0), 2)



cv2.imshow("elon image", imgElon)
cv2.imshow("test image", imgTest)

cv2.waitKey(0)