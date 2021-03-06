# Copyright 2018
# Kien Nguyen
# k.nguyenthanh@qut.edu.au

import cv2
import sys
import numpy as np


if __name__ == '__main__':

  faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
  smileCascade = cv2.CascadeClassifier('models/haarcascade_smile.xml')
  smileNeighborsMax = 100
  neighborStep = 2

  frame = cv2.imread("hillary_clinton.jpg")
  #frame = cv2.imread("people.jpg")

  frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(frameGray, 1.2, 3)
  for (x, y, w, h) in faces:

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    faceRoiGray = frameGray[y: y + h, x: x + w]
    faceRoiOriginal = frame[y: y + h, x: x + w]

    for neigh in range(1, smileNeighborsMax, neighborStep):
      smile = smileCascade.detectMultiScale(faceRoiGray, 1.5, neigh)

      frameClone = np.copy(frame)
      faceRoiClone = frameClone[y: y + h, x: x + w]
      for (xx, yy, ww, hh) in smile:
        cv2.rectangle(faceRoiClone, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)

      cv2.putText(frameClone, "# Neighbors = {}".format(neigh), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
      cv2.imshow('Face and Smile Demo', frameClone)
      if cv2.waitKey(500) & 0xFF == 27:
        cv2.destroyAllWindows()
        sys.exit()