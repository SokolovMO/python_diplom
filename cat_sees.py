import cv2 as cv2
import numpy as np

cap = cv2.VideoCapture(0) # создание экземпляра класса для захвата видео со стереокамеры
frame_counter = 16 #счётчик сохранённых кадров

#определение размера изображения с камеры

ret, img = cap.read() #ret - статус чтения с камеры (true|false) img - изображение с камеры
height, width, _ = np.array(img).shape #получили размер стереограммы
width//=2 #один кадра в два раза уже

#получение видео с камеры
#захват левых/правых кадров со стереокмеры (с шаблоном для калибровки)

while cv2.waitKey(1) != ord('q'): #пока не нажата q
    ret, img = cap.read() #ret - статус чтения с камеры (true|false) img - изображение с камеры
    imgR = img[:,:width,:] #правая часть стереограммы
    imgL = img[:,width:,:] #левая часть стереограммы
    cv2.imshow("left", imgL) #вывод изображения 
    cv2.imshow("right", imgR) #вывод изображения 
    if cv2.waitKey(1) == ord('c'): #если нажата с
        frame_counter += 1 #инеремент счётчика сохранённых кадров
        cv2.imwrite('./pairs/catch_left_'+str(frame_counter)+'.png', imgL) #сохранение кадра 
        cv2.imwrite('./pairs/catch_right_'+str(frame_counter)+'.png', imgR) #сохранение кадра 

cap.release()
cv2.destroyAllWindows()
