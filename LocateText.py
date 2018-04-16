#!/usr/bin/python
# vim: set ts=2 expandtab:
#import clean_page as clean
import connected_components as cc
import run_length_smoothing as rls
import clean_page as clean
import ocr
import segmentation as seg
import furigana
import arg
import defaults
from scipy.misc import imsave

import numpy as np
import cv2
import sys
import os
import scipy.ndimage
import glob

if __name__ == '__main__':

  AbsPath = 'C:/Users/blue.i/Desktop/New folder/MangaTextDetection-master/'
  for filename in glob.glob(os.path.join(AbsPath, '*.jpg')):
      infile = ""+filename
      print(infile)
      outfile = infile + '.text_areas.jpg'
      img = cv2.imread(infile)
      gray = clean.grayscale(img)

      binary_threshold=arg.integer_value('binary_threshold',default_value=defaults.BINARY_THRESHOLD)
      if arg.boolean_value('verbose'):
        print ('Binarizing with threshold value of ' + str(binary_threshold))
      inv_binary = cv2.bitwise_not(clean.binarize(gray, threshold=binary_threshold))
      binary = clean.binarize(gray, threshold=binary_threshold)

      segmented_image = seg.segment_image(gray)
      segmented_image = segmented_image[:,:,2]
      myImage = np.copy(segmented_image)
      image, contours, hierarchy = cv2.findContours(myImage,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
      for cnt in contours:
          if cv2.contourArea(cnt) > 4000 :
              cv2.drawContours(img,[cnt],0,(255,255,255),cv2.FILLED)
      imsave(outfile,img)
