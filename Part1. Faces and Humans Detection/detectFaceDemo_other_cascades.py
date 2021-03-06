# Copyright 2018
# Kien Nguyen
# k.nguyenthanh@qut.edu.au

import cv2
import sys
import numpy as np

if __name__ == '__main__':

#  faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
  faceCascade = cv2.CascadeClassifier('models/lbpcascade_frontalface.xml')

# HOG no longer supported
#  faceCascade = cv2.CascadeClassifier('models/hogcascade_pedestrians.xml')

  faceNeighborsMax = 10
  neighborStep = 1

  frame = cv2.imread("test3.jpg")


  frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  for neigh in range(1, faceNeighborsMax, neighborStep):
    faces = faceCascade.detectMultiScale(frameGray, 1.1, neigh)
    frameClone = np.copy(frame)

    for (x, y, w, h) in faces:
      cv2.rectangle(frameClone, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.putText(frameClone, "# Neighbors = {}".format(neigh), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    cv2.imshow('Face Detection Demo', frameClone)
    if cv2.waitKey(500) & 0xFF == 27:
      cv2.destroyAllWindows()
      sys.exit()