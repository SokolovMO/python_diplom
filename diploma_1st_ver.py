# import cv2
# import numpy as np
#
#
# capture = cv2.VideoCapture(0)
# while True:
#     ret, img = capture.read()
#     cv2.imshow('From Camera', img)
#
#     k = cv2.waitKey(30) & 0xFF
#     if k == 27:
#         break
#
# capture.release ()
# cv2.destroyAlLWindows()
import cv2
import numpy as np
from scipy.stats import itemfreq

# Функция использует алгоритм кластеризации K-mean, чтобы найти доминирующий цвет
def get_dominant_color(image, n_colors):
    # Изменение формы массива изображений в 2D-массив из 3-х кортежей (представляющих значения RGB)
    pixels = np.float32(image).reshape((-1, 3))
    # Определение критерия для алгоритма кластеризации K-mean
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    # Уточнение критерия для использования случайных центров на изображении
    flags = cv2.KMEANS_RANDOM_CENTERS
    # Применение кластеризации к значениям RGB пикселей
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    # Возврат значения RGB наиболее распространенного кластера
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]

# Эта функция устанавливает значение глобальной переменной "clicked" в True
# при возникновении события левой кнопкой мыши.
# Это используется для остановки цикла обработки в основном коде.
clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

# Запуск трансляции с камеры
cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('webcam')
cv2.setMouseCallback('webcam', onMouse)

#  Считывание и обработка кадров в цикле
success, frame = cameraCapture.read()



# В этой части при захвате изображения с камеры происходит поиск круга в кадрах
while success and not clicked:
    cv2.waitKey(1)
    success, frame = cameraCapture.read()
    # Преобразование в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Медианное размытие для уменьшение шума
    img = cv2.medianBlur(gray, 37)
    # Обнаружение кругов
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                              1, 50, param1=120, param2=40)

    # Если обнаружен хотя бы один круг
    if not circles is None:
        # Округление координат круга до ближайшего целого числа
        circles = np.uint16(np.around(circles))
        # Для поиска наибольшего круга
        max_r, max_i = 0, 0
        # Цикл для всех обнаруженных кругов
        for i in range(len(circles[:, :, 2][0])):
            if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                max_i = i
                max_r = circles[:, :, 2][0][i]
        # Координаты и радиус наибольшего круга
        x, y, r = circles[:, :, :][0][max_i]
        if y > r and x > r:
            # Создание области в виде квадрата для самого большого круга
            square = frame[y-r:y+r, x-r:x+r]
            # Доминирующий цвет в выбранной области. В OpenCV BGR, 0 - blue, 1 - green, 2 - red
            dominant_color = get_dominant_color(square, 2)
            # Если доминирует красный, то из 6 знаков это будет знак СТОП
            if dominant_color[2] > 100:
                print("стоп")
            # Если доминирующий цвет синий, то остается определить направление одной или двух одновременно стрелок
            elif dominant_color[0] > 80:
                zone_0 = square[square.shape[0]*3//8:square.shape[0]
                                * 5//8, square.shape[1]*1//8:square.shape[1]*3//8]
                zone_0_color = get_dominant_color(zone_0, 1)

                zone_1 = square[square.shape[0]*1//8:square.shape[0]
                                * 3//8, square.shape[1]*3//8:square.shape[1]*5//8]
                zone_1_color = get_dominant_color(zone_1, 1)

                zone_2 = square[square.shape[0]*3//8:square.shape[0]
                                * 5//8, square.shape[1]*5//8:square.shape[1]*7//8]
                zone_2_color = get_dominant_color(zone_2, 1)

                if zone_1_color[2] < 60:
                    if sum(zone_0_color) > sum(zone_2_color):
                        print("налево")
                    else:
                        print("направо")
                else:
                    if sum(zone_1_color) > sum(zone_0_color) and sum(zone_1_color) > sum(zone_2_color):
                        print("прямо")
                    elif sum(zone_0_color) > sum(zone_2_color):
                        print("прямо или налево")
                    else:
                        print("прямо или направо")
            # Если программа не может распознать знак, то она обязательно об этом сообщит
            else:
                print("не распознано")
        # Этот цикл демонстирует распозанные круги, обнаруженные с помощью HoughCircles на рамке
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    # Демонастрация изображения на экран с веб-камеры и остановка демонстации на клавишу esc
    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cv2.destroyAllWindows()
cameraCapture.release()