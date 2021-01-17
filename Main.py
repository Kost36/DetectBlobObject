import cv2 as cv
import numpy as np
import pathlib as pathlib
from math import *

#Переменные
dir=pathlib.Path.cwd().__str__()+"\img" #Путь к папке проекта
thresholdSizeMul=2.5 #Порог отклонения размера от среднего размера
thresholdDistToCenterMul=0.2 #Порог отклонения центра блоба от центров по осям X и Y

#Расстояние между двумя точками
def Distance(x1, y1, x2, y2):
    c = sqrt((x2-x1)**2 + (y2-y1)**2)
    return c

#Проверка позиций блобов
def CheckPozitionBlobs(keypoints, xCentr, yCentr):
    sumDistantionOfCenter=0 #Сумма растояний до центра image
    sumSize=0 #Сумма размеров blob ов

    #Считаем суммы
    for keypoint in keypoints:
        sumDistantionOfCenter+=Distance(keypoint.pt[0], keypoint.pt[1], xCentr, yCentr) #Сумма растояния до центра
        sumSize+=keypoint.size #Сумма размеров блобов

    #Среднее
    averDistantionOfCenter=sumDistantionOfCenter/4.0  #Среднее растояний до центра image
    averSize=sumSize/4.0 #Средний размер блобов

    zonesOk = np.array([False, False, False, False]) #Массив решений по зонам
    #Распределение блобов по зонам
    for keypoint in keypoints:
        if ((keypoint.pt[0]<xCentr) & (keypoint.pt[1]<yCentr)): #Если blob в зоне
            zonesOk[0]=True #В зоне есть объект
        if ((keypoint.pt[0]>=xCentr) & (keypoint.pt[1]<yCentr)): #Если blob в зоне
            zonesOk[1]=True #В зоне есть объект
        if ((keypoint.pt[0]<xCentr) & (keypoint.pt[1]>=yCentr)): #Если blob в зоне
            zonesOk[2]=True #В зоне есть объект
        if ((keypoint.pt[0]>=xCentr) & (keypoint.pt[1]<yCentr)): #Если в blob зоне
            zonesOk[3]=True #В зоне есть объект

    #Проверка заполнения всех зон блобами
    for boolResult in zonesOk:
        if (boolResult): #Если в зоне есть блоб
            pass
        else: #Если в зоне нету блоба
            return False

    #Проверка приблизительно равного размера блобов
    for keypoint in keypoints:
        if ((keypoint.size>averSize+averSize*thresholdSizeMul) | (keypoint.size<averSize-averSize*thresholdSizeMul)):
            return False #Если размер блоба выходит за порог, то вернем нет

    #Пороги для фильтра по расстоянию блоба до центра Image
    thresholdDistToCenterMin=averDistantionOfCenter-averDistantionOfCenter*thresholdDistToCenterMul
    thresholdDistToCenterMax=averDistantionOfCenter+averDistantionOfCenter*thresholdDistToCenterMul

    #Фильтр по расстоянию от центра Image до центров blobs
    for keypoint in keypoints:
        distance=Distance(keypoint.pt[0], keypoint.pt[1], xCentr, yCentr) #Рассчет дистанции до центра
        if ((distance<thresholdDistToCenterMin) | (distance>thresholdDistToCenterMax)):
            return False #Если расстоянию от центра Image до центров blob выходит за порог, то вернем нет

    return True
    pass

# Проверка блобов
def CheckBlobs(keypoints, xCentr, yCentr):
    if (len(keypoints)!=4): # Должно быть 4 шт.
        return False
    if (CheckPozitionBlobs(keypoints, xCentr, yCentr) == False): # Проверка позиций блобов.
        return False
    return True
    pass

# Поиск Blobs
def SearchBlobs(image):
    # Параметры поиска Blob-ов
    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 250  # Без шагов, т.к уже бинаризованных вход
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 20000
    params.filterByCircularity = True
    params.minCircularity = 0.6  # Увеличение приводит к NoSearch 4 блобов на 11 image
    params.maxCircularity = 1
    params.filterByConvexity = True
    params.minConvexity = 0.85  # Увеличение приводит к NoSearch 4 блобов на 11 image
    params.maxConvexity = 1
    params.filterByInertia = False
    params.filterByColor = False
    params.minDistBetweenBlobs = 30  # Фильтр лишнего

    detector = cv.SimpleBlobDetector_create(params)  # Создадим детектор блобов
    keypoints = detector.detect(image)  # Поиск блобов
    return keypoints
    pass

#Обработка изображения
def ProcessingImage(imageInput, numberImage):
    # Получаем центра по X и Y
    xCentr = imageInput.shape[1]/2.0 #Центр по оси X
    yCentr = imageInput.shape[0]/2.0 #Центр по оси Y

    # Бинаризация порог = 175
    # Уменьшение приводит к слиянию интересующих блобов
    # Увеличение уменьшает размер слабого блоба на 11 image
    ret, imageBinary = cv.threshold(imageInput, 175, 255, cv.THRESH_BINARY)

    # Уберем возможные шумы
    kernel = np.ones((5, 5), 'uint8')
    imageErode = cv.erode(imageBinary, kernel, iterations=1) #Убрать шумы
    imageDilate = cv.dilate(imageErode, kernel, iterations=3) #Вернем,и увеличим размер

    keypoints = SearchBlobs(imageDilate) #Поиск блобов

    # Нанесение интересующих областей(зоны)
    imageOutput = cv.rectangle(imageInput,
                               (0, 0), (int(xCentr), int(yCentr)), (0, 255, 255), 2)
    imageOutput = cv.rectangle(imageOutput,
                               (int(xCentr), 0), (int(xCentr) * 2, int(yCentr)), (0, 255, 255), 2)
    imageOutput = cv.rectangle(imageOutput,
                               (0, int(yCentr)), (int(xCentr), int(yCentr) * 2), (0, 255, 255), 2)
    imageOutput = cv.rectangle(imageOutput,
                               (int(xCentr), int(yCentr)), (int(xCentr) * 2, int(yCentr) * 2), (0, 255, 255), 2)

    # Нанесение найденных объектов
    imageWithKeypoints = cv.drawKeypoints(imageOutput, keypoints, np.array([]), (0, 0, 255),
                                          cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Нанесение результата обработки image
    if (CheckBlobs(keypoints, xCentr, yCentr)): #Если проверка пройдена
        imageWithKeypoints = cv.putText(imageWithKeypoints, 'Yes', #Image, текст
                                        (10, imageInput.shape[0]-10), #Позиция (X,Y начала текста)
                                        cv.FONT_HERSHEY_SIMPLEX, 2, #Тип фона, множитель размера
                                        (0, 255, 0), 5) #Цвет фона, толщина линий
        print("Image " + numberImage.__str__() + " Yes")
    else: #Если проверка не пройдена
        imageWithKeypoints = cv.putText(imageWithKeypoints, 'No', #Image, текст
                                        (10, imageInput.shape[0]-10), #Позиция (X,Y начала текста)
                                        cv.FONT_HERSHEY_SIMPLEX, 2, #Тип фона, множитель размера
                                        (0, 0, 255), 5) #Цвет фона, толщина линий
        print("Image " + numberImage.__str__() + " No")

    return imageWithKeypoints
    pass

# Догика программы
for i in range(15): # Пробегаемся по 14 цифрам -> номера images
    filePath = dir + '\\'.__str__() + i.__str__() #Путь к image
    image=cv.imread(filePath + ".bmp", cv.IMREAD_COLOR) #Грузим image в формате bmp
    if (image is None): #Если не загрузилось
        image = cv.imread(filePath + ".png", cv.IMREAD_COLOR)  #Грузим image в формате png
    outputImage=ProcessingImage(image, i) #Обработка image
    cv.imshow("Image " + i.__str__() + ". For next click exit", outputImage) #Вывод результата
    cv.imwrite(filePath + "_Result.png", outputImage)
    btn = input()
    #btn=cv.waitKey(0)