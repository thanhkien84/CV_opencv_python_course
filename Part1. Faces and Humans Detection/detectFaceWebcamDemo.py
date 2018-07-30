# Copyright 2018
# Kien Nguyen
# k.nguyenthanh@qut.edu.au


import cv2
import sys
import numpy as np

if __name__ == '__main__':

  camera = cv2.VideoCapture(0)

#  faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
  faceCascade = cv2.CascadeClassifier('models/haarcascade_upperbody.xml')

  while (True):
    ret, frame = camera.read()

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(frameGray, 1.1, 10)
    frameClone = np.copy(frame)

    for (x, y, w, h) in faces:
      cv2.rectangle(frameClone, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Detection Demo', frameClone)
    if cv2.waitKey(500) & 0xFF == 27:
      cv2.destroyAllWindows()
      sys.exit()