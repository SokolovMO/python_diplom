import os
import cv2
import numpy as np
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

#параметры кадра

img = cv2.imread('./pairs/catch_left_1.png') #прочитали кадр для калибровки
height, width, _ = np.array(img).shape #получили размер кадра
image_size = (width, height) #получили размер кадра

#параметры доски

rows = 4 #количество персечений клеток, строк
columns = 6 #количество персечений клеток, столбцов
square_size = 3        #?????????????????????                                            

# калибровка

calibrator = StereoCalibrator(rows, columns, square_size, image_size)
n_images = 45  #количество пар для калибровки
for i in range(1, n_images+1): #цикл по парам
    print ('taking pair ' + str(i)) #сообщение о начале работы с парой
    name_left = './pairs/catch_left_'+str(i)+'.png' #имя левой половинки
    name_right = './pairs/catch_right_'+str(i)+'.png' #имя правой половинки
    # if os.path.isfile(name_left) and os.path.isfile(name_right): #если обе половинки существуют
    image_left = cv2.imread(name_left) #читаем левую половинку
    image_right = cv2.imread(name_right) #читаем правую половинку
    try: # определяем углы доски
        calibrator._get_corners(image_left)
        calibrator._get_corners(image_right)
    except ChessboardNotFoundError as error: #если на кадрах не найдена доска
        print ("pair "+ str(i) + ", where is my chessboard?")
    else:
        calibrator.add_corners((image_left, image_right), True)      
print ('End cycle')


print ('Starting calibration... It can take several minutes!')
calibration = calibrator.calibrate_cameras()
calibration.export('calib_result')
print ('Calibration complete!')


# Lets rectify and show last pair after  calibration
calibration = StereoCalibration(input_folder='calib_result')
rectified_pair = calibration.rectify((image_left, image_right))
  
cv2.imshow('Left CALIBRATED', rectified_pair[0])
cv2.imshow('Right CALIBRATED', rectified_pair[1])
cv2.imwrite("rectifyed_left.jpg",rectified_pair[0])
cv2.imwrite("rectifyed_right.jpg",rectified_pair[1])
cv2.waitKey(0)