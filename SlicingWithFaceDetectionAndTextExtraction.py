import cv2
import numpy as np
import glob
import os
import math
import connected_components as cc
import run_length_smoothing as rls
import clean_page as clean
import ocr
import segmentation as seg
import furigana
import arg
import defaults
from scipy.misc import imsave
import sys
import scipy.ndimage
AbsPath = 'D:/semesters/graduation project - manga/Manga109/Manga109/images'
#AbsPath = 'D:/semesters/graduation project - manga/TennenSenshiG'
cascPath = "D:/semesters/graduation project - manga/lbpcascade_animeface.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
MangaCounter = -1
FeaturesFile = open('D:/semesters/graduation project - manga/features.csv', 'w+')
MangaFile = open('D:/semesters/graduation project - manga/mangaNames.csv', 'w+')
MangaFile.write(",MangaName\n")
FeaturesFile.write(",f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30\n")
for mangaName in os.listdir(AbsPath):
    print(mangaName)
    MangaCounter += 1
    MangaFile.write(str(MangaCounter) + "," + mangaName+"\n")
    FeaturesFileString = "";
    path = AbsPath+"/"+mangaName
    counter = 0
    counterRatio = 0
    counterRatioOnly=0
    faceCounter = 0
    mangaFacesFeatures = []
    mangaScenesFeatures = []
    facePixelsValuesCounter = 0
    scenePixelsValuesCounter = 0
    facePixelsValuesFreq = np.zeros((17,), dtype=int)
    scenePixelsValuesFreq = np.zeros((17,), dtype=int)
    for filename in glob.glob(os.path.join(path, '*.jpg')):
        facesVector = []
        pageVector = []
        outLinesPixels = []
        tonePixels = []
        def Scene_Pixels_Values_Freq( Picture ):
            for row in Picture:
                for pixel in row:
                    if pixel>40 and pixel<=210  :
                        scenePixelsValuesFreq[int((pixel-41)/10)]+=1
        def Face_Pixels_Values_Freq( Picture ):
            for row in Picture:
                for pixel in row:
                    if pixel>40 and pixel<=210  :
                        facePixelsValuesFreq[int((pixel-41)/10)]+=1
        def Feature_Page_Average_Width( vectorWidth ):
            if len(vectorWidth)!=0 :
                averageWidth = 0;
                for width in vectorWidth:
                    averageWidth += width
                if len(vectorWidth)!=0 :
                    averageWidth /= len(vectorWidth)
                pageVector.append(averageWidth)
        def Feature_Page_Average_Height( vectorHeight ):
            if len(vectorHeight)!=0 :
                averageHeight = 0;
                for Height in vectorHeight:
                    averageHeight += Height
                if len(vectorHeight)!=0 :
                    averageHeight /= len(vectorHeight)
                pageVector.append(averageHeight)
        def Feature_Page_Average_Slope( vectorPoints ):
            if len(vectorPoints)!=0 :
                averageDivertSlope = 1;
                counter = 0;
                for points in vectorPoints:
                    Xs = []
                    Ys = []
                    for point in points:
                        Xs.append(point[0][0])
                        Ys.append(point[0][1])
                    for i in range(0, 4):
                        if not(abs(Xs[i]-Xs[(i+1)%4])<10 or abs(Ys[i]-Ys[(i+1)%4])<10) :
                            counter+=1
                            averageDivertSlope += ((abs(Ys[i]-Ys[(i+1)%4]))/(abs(Xs[i]-Xs[(i+1)%4])))
                if counter != 0 :
                    averageDivertSlope /= counter
                pageVector.append(averageDivertSlope)
        def Feature_Page_Average_Area_To_Rectangle_Ratio( vectorArea ):
            if len(vectorArea)!=0 :
                averageArea = 0;
                for Area in vectorArea:
                    averageArea+= Area
                if len(vectorArea)!=0 :
                    averageArea /= len(vectorArea)
                pageVector.append(averageArea)
        def Feature_Page_Average_Canny_Lines(vectorCannyLines,CannyAreas):
            if len(vectorCannyLines)!=0 :
                averageCannyLines = 0;
                counter = 0
                CannyAreas[counter] +=1
                for CannyLines in vectorCannyLines:
                    try:
                        averageCannyLines+= CannyLines/CannyAreas[counter]
                    except ZeroDivisionError:
                        averageCannyLines+= 0
                    counter +=1
                if len(vectorCannyLines)!=0:
                    averageCannyLines /= len(vectorCannyLines)
                pageVector.append(averageCannyLines)
        def Feature_SceneCanny_Pixels(Picture,Area):
            counter = 0
            for row in Picture:
                for pixel in row:
                    if pixel!=0 :
                        counter+=1
            outLinesPixels.append(counter/Area)
        def Feature_SceneTone_Pixels(Picture,Area):
            counter = 0
            for row in Picture:
                for pixel in row:
                    if pixel<220 :
                        counter+=1
            tonePixels.append(counter/Area)
        def Feature_SceneCanny_Average_Pixels():
            if len(outLinesPixels)!=0 :
                average = 0
                for Pixels in outLinesPixels:
                    average+=Pixels
                if len(outLinesPixels)!=0:
                    average /= len(outLinesPixels)
                pageVector.append(average)
        def Feature_Tone_Average_Pixels():
            if len(tonePixels)!=0 :
                average = 0
                for Pixels in tonePixels:
                    average+=Pixels
                if len(tonePixels)!=0:
                    average /= len(tonePixels)
                pageVector.append(average)
        def Feature_Outlines_To_Tones_Pixels_Ratio():
            if len(outLinesPixels)!=0 :
                average = 0
                counter = 0
                if tonePixels[counter] == 0 :
                    tonePixels[counter] +=1
                for Pixels in outLinesPixels:
                    try:
                        average+=Pixels/tonePixels[counter]
                    except ZeroDivisionError:
                        average+=0
                    counter += 1
                if len(outLinesPixels) != 0 :
                    average /= len(outLinesPixels)
                pageVector.append(average)
        def Feature_FaceCanny_Pixels(Picture,Area):
            counter = 0
            for row in Picture:
                for pixel in row:
                    if pixel!=0 :
                        counter+=1
            return counter/Area
        def Feature_FaceTone_Pixels(Picture,Area):
            counter = 0
            for row in Picture:
                for pixel in row:
                    if pixel<220 :
                        counter+=1
            return counter/Area
        img = cv2.imread(filename)
        source_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        row, col= source_gray.shape[:2]
        bottom= source_gray[row-2:row, 0:col]
        mean= cv2.mean(bottom)[0]
        bordersize = 3
        border = cv2.copyMakeBorder(source_gray, top=bordersize, bottom=bordersize, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
        bordersize = 2
        borderWhite = cv2.copyMakeBorder(border, top=bordersize, bottom=bordersize, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
        ret,source_thresh = cv2.threshold(borderWhite,230,255,0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2) ,(1, 1))
        source_dilated = cv2.dilate(source_thresh, kernel, iterations=1)

        kernel_size = 3
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        gray_lap = cv2.Laplacian(source_dilated,ddepth,ksize = kernel_size,scale = scale,delta = delta)
        dst = cv2.convertScaleAbs(gray_lap)
        im2, contours, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(img, contours, -1, (0,255,0), 3)
        # Find the index of the largest contour
        pageFacesWidths = []
        faces = faceCascade.detectMultiScale(
            borderWhite,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(100, 100)
            )
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            faceFeatures = []
            crop_face_img = borderWhite[y:y+h, x:x+w]
            #cv2.imwrite(pathFaces+"/"+str(faceCounter)+".jpg", crop_face_img)
            faceCounter += 1
            faceFeatures.append(w*h)
            edges = cv2.Canny(crop_face_img,500,500)
            edgesOfFaces = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            faceFeatures.append(len(edgesOfFaces[1])/(w*h)) # density per unit area
            faceFeatures.append(Feature_FaceCanny_Pixels(edges,w*h)) #faceCanny Pixels per unit area
            FaceTonePicture = cv2.bitwise_or(crop_face_img, edges)   # faceTone preprocessing
            faceFeatures.append(Feature_FaceTone_Pixels(FaceTonePicture,w*h))
            faceFeatures.append(Feature_FaceCanny_Pixels(edges,w*h)/Feature_FaceTone_Pixels(FaceTonePicture,w*h))
            facesVector.append(faceFeatures)
            Face_Pixels_Values_Freq(crop_face_img)
        #Feature_Page_Average_Faces_Areas(pageFacesWidths)


        width, height = borderWhite.shape
        subRegions = []
        areas = [cv2.contourArea(c) for c in contours]
        widths= []
        heights= []
        AreaToRectangleRatio= []
        Points= []
        CannyLines = []
        CannyAreas = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 20000 :
                epsilon = 0.01*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                if len(approx) == 4 :
                    mask = np.zeros([width, height, 3], dtype = "uint8")
                    mask_gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                    cv2.drawContours(mask_gray,[cnt],0,(255,255,255),cv2.FILLED)
                    mask_gray_white = cv2.bitwise_not(mask_gray)
                    masked_img = cv2.bitwise_and(borderWhite, mask_gray, mask)
                    masked_img = cv2.bitwise_or(mask_gray_white, masked_img)
                    x,y,w,h = cv2.boundingRect(cnt)
                    crop_img = masked_img[y:y+h, x:x+w]
                    binary_threshold=arg.integer_value('binary_threshold',default_value=defaults.BINARY_THRESHOLD)
                    if arg.boolean_value('verbose'):
                      print ('Binarizing with threshold value of ' + str(binary_threshold))
                    inv_binary = cv2.bitwise_not(clean.binarize(crop_img, threshold=binary_threshold))
                    binary = clean.binarize(crop_img, threshold=binary_threshold)

                    segmented_image = seg.segment_image(crop_img)
                    segmented_image = segmented_image[:,:,2]
                    mySceneImage = np.copy(segmented_image)
                    image, contours, hierarchy = cv2.findContours(mySceneImage,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cv2.contourArea(cnt) > 3000 :
                            cv2.drawContours(crop_img,[cnt],0,(255,255,255),cv2.FILLED)
                    #cv2.imshow('image',crop_img)
                    #cv2.waitKey(0)
                    #cv2.imwrite(pathSolved+"/"+str(counter)+".jpg", crop_img)
                    edges = cv2.Canny(crop_img,500,500)
                    edgesContours = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                    CannyLines.append(len(edgesContours[1]))
                    widths.append(w)
                    heights.append(h)
                    AreaToRectangleRatio.append(cv2.contourArea(cnt)/(w*h))
                    Points.append(approx)
                    CannyAreas.append(cv2.contourArea(cnt))
                    Feature_SceneCanny_Pixels(edges,cv2.contourArea(cnt))
                    TonePicture = cv2.bitwise_or(crop_img, edges)
                    Feature_SceneTone_Pixels(TonePicture,cv2.contourArea(cnt))
                    Scene_Pixels_Values_Freq(crop_img)
                    counter+=1
        Feature_Page_Average_Width(widths)
        Feature_Page_Average_Height(heights)
        Feature_Page_Average_Area_To_Rectangle_Ratio(AreaToRectangleRatio)
        Feature_Page_Average_Slope(Points)
        Feature_Page_Average_Canny_Lines(CannyLines,CannyAreas)
        Feature_SceneCanny_Average_Pixels()
        Feature_Tone_Average_Pixels()
        Feature_Outlines_To_Tones_Pixels_Ratio()
        #print("pv"+str(pageVector)+"     fv"+str(facesVector))
        mangaFacesFeatures.append(facesVector)
        mangaScenesFeatures.append(pageVector)

    MangaFaceFeaturesAverage = [0]*5
    MangaFaceFeaturesdaviation = [0]*5
    for FacesPerPage in mangaFacesFeatures:
        for face in FacesPerPage:
            if len(face)>0 :
                MangaFaceFeaturesAverage[0] += face[0]
                MangaFaceFeaturesAverage[1] += face[1]
                MangaFaceFeaturesAverage[2] += face[2]
                MangaFaceFeaturesAverage[3] += face[3]
                MangaFaceFeaturesAverage[4] += face[4]
    MangaFaceFeaturesAverage[0] /= faceCounter+1
    MangaFaceFeaturesAverage[1] /= faceCounter+1
    MangaFaceFeaturesAverage[2] /= faceCounter+1
    MangaFaceFeaturesAverage[3] /= faceCounter+1
    MangaFaceFeaturesAverage[4] /= faceCounter+1
    for FacesPerPage in mangaFacesFeatures:
        for face in FacesPerPage:
            if len(face)>0 :
                MangaFaceFeaturesdaviation[0] += (MangaFaceFeaturesAverage[0]-face[0])*(MangaFaceFeaturesAverage[0]-face[0])
                MangaFaceFeaturesdaviation[1] += (MangaFaceFeaturesAverage[1]-face[1])*(MangaFaceFeaturesAverage[1]-face[1])
                MangaFaceFeaturesdaviation[2] += (MangaFaceFeaturesAverage[2]-face[2])*(MangaFaceFeaturesAverage[2]-face[2])
                MangaFaceFeaturesdaviation[3] += (MangaFaceFeaturesAverage[3]-face[3])*(MangaFaceFeaturesAverage[3]-face[3])
                MangaFaceFeaturesdaviation[4] += (MangaFaceFeaturesAverage[4]-face[4])*(MangaFaceFeaturesAverage[4]-face[4])
    MangaFaceFeaturesdaviation[0] = math.sqrt(MangaFaceFeaturesdaviation[0] / (faceCounter+1))/MangaFaceFeaturesAverage[0]
    MangaFaceFeaturesdaviation[1] = math.sqrt(MangaFaceFeaturesdaviation[1] / (faceCounter+1))/MangaFaceFeaturesAverage[1]
    MangaFaceFeaturesdaviation[2] = math.sqrt(MangaFaceFeaturesdaviation[2] / (faceCounter+1))/MangaFaceFeaturesAverage[2]
    MangaFaceFeaturesdaviation[3] = math.sqrt(MangaFaceFeaturesdaviation[3] / (faceCounter+1))/MangaFaceFeaturesAverage[3]
    MangaFaceFeaturesdaviation[4] = math.sqrt(MangaFaceFeaturesdaviation[4] / (faceCounter+1))/MangaFaceFeaturesAverage[4]

    MangaSceneFeaturesAverage = [0]* 8
    MangaSceneFeaturesdaviation = [0]* 8
    for scene in mangaScenesFeatures:
        if len(scene)>0 :
            MangaSceneFeaturesAverage[0] += scene[0]
            MangaSceneFeaturesAverage[1] += scene[1]
            MangaSceneFeaturesAverage[2] += scene[2]
            MangaSceneFeaturesAverage[3] += scene[3]
            MangaSceneFeaturesAverage[4] += scene[4]
            MangaSceneFeaturesAverage[5] += scene[5]
            MangaSceneFeaturesAverage[6] += scene[6]
            MangaSceneFeaturesAverage[7] += scene[7]
    MangaSceneFeaturesAverage[0] /= counter+1
    MangaSceneFeaturesAverage[1] /= counter+1
    MangaSceneFeaturesAverage[2] /= counter+1
    MangaSceneFeaturesAverage[3] /= counter+1
    MangaSceneFeaturesAverage[4] /= counter+1
    MangaSceneFeaturesAverage[5] /= counter+1
    MangaSceneFeaturesAverage[6] /= counter+1
    MangaSceneFeaturesAverage[7] /= counter+1
    for scene in mangaScenesFeatures:
        if len(scene)>0 :
            MangaSceneFeaturesdaviation[0] += (MangaSceneFeaturesAverage[0]-scene[0])*(MangaSceneFeaturesAverage[0]-scene[0])
            MangaSceneFeaturesdaviation[1] += (MangaSceneFeaturesAverage[1]-scene[1])*(MangaSceneFeaturesAverage[1]-scene[1])
            MangaSceneFeaturesdaviation[2] += (MangaSceneFeaturesAverage[2]-scene[2])*(MangaSceneFeaturesAverage[2]-scene[2])
            MangaSceneFeaturesdaviation[3] += (MangaSceneFeaturesAverage[3]-scene[3])*(MangaSceneFeaturesAverage[3]-scene[3])
            MangaSceneFeaturesdaviation[4] += (MangaSceneFeaturesAverage[4]-scene[4])*(MangaSceneFeaturesAverage[4]-scene[4])
            MangaSceneFeaturesdaviation[5] += (MangaSceneFeaturesAverage[5]-scene[5])*(MangaSceneFeaturesAverage[5]-scene[5])
            MangaSceneFeaturesdaviation[6] += (MangaSceneFeaturesAverage[6]-scene[6])*(MangaSceneFeaturesAverage[6]-scene[6])
            MangaSceneFeaturesdaviation[7] += (MangaSceneFeaturesAverage[7]-scene[7])*(MangaSceneFeaturesAverage[7]-scene[7])
    MangaSceneFeaturesdaviation[0] = math.sqrt(MangaSceneFeaturesdaviation[0] / (counter+1))/MangaSceneFeaturesAverage[0]
    MangaSceneFeaturesdaviation[1] = math.sqrt(MangaSceneFeaturesdaviation[1] / (counter+1))/MangaSceneFeaturesAverage[1]
    MangaSceneFeaturesdaviation[2] = math.sqrt(MangaSceneFeaturesdaviation[2] / (counter+1))/MangaSceneFeaturesAverage[2]
    MangaSceneFeaturesdaviation[3] = math.sqrt(MangaSceneFeaturesdaviation[3] / (counter+1))/MangaSceneFeaturesAverage[3]
    MangaSceneFeaturesdaviation[4] = math.sqrt(MangaSceneFeaturesdaviation[4] / (counter+1))/MangaSceneFeaturesAverage[4]
    MangaSceneFeaturesdaviation[5] = math.sqrt(MangaSceneFeaturesdaviation[5] / (counter+1))/MangaSceneFeaturesAverage[5]
    MangaSceneFeaturesdaviation[6] = math.sqrt(MangaSceneFeaturesdaviation[6] / (counter+1))/MangaSceneFeaturesAverage[6]
    MangaSceneFeaturesdaviation[7] = math.sqrt(MangaSceneFeaturesdaviation[7] / (counter+1))/MangaSceneFeaturesAverage[7]
    numberOfGradientColorsInFaces = 0
    numberOfGradientColorsInScenes = 0
    mostGradientColorInFaces = -1
    mostGradientColorInScene = -1
    counterGradientFaces = 0
    counterGradientScenes = 0
    for feature in facePixelsValuesFreq:
        facePixelsValuesCounter += feature
    for feature in scenePixelsValuesFreq:
        scenePixelsValuesCounter += feature
    for feature in facePixelsValuesFreq:
        if mostGradientColorInFaces < feature :
            mostGradientColorInFaces = counterGradientFaces
        compareValue = (feature/facePixelsValuesCounter)
        if compareValue >= 0.07 :
            numberOfGradientColorsInFaces +=1
        counterGradientFaces += 1
    for feature in scenePixelsValuesFreq:
        if mostGradientColorInScene < feature :
            mostGradientColorInScene = counterGradientScenes
        compareValue = (feature/scenePixelsValuesCounter)
        if compareValue >= 0.07 :
            numberOfGradientColorsInScenes +=1
        counterGradientScenes += 1
    FeaturesFileString += str(MangaCounter)
    for feature in MangaSceneFeaturesdaviation:
        FeaturesFileString +=  "," + str(feature)
    for feature in MangaSceneFeaturesAverage:
        FeaturesFileString +=  "," + str(feature)
    for feature in MangaFaceFeaturesdaviation:
        FeaturesFileString +=  "," + str(feature)
    for feature in MangaFaceFeaturesAverage:
        FeaturesFileString +=  "," + str(feature)
    FeaturesFileString +=  "," + str(numberOfGradientColorsInFaces) +  "," + str(numberOfGradientColorsInScenes) +  "," + str(mostGradientColorInFaces) +  "," + str(mostGradientColorInScene)
    FeaturesFileString +="\n"
    FeaturesFile.write(FeaturesFileString)
