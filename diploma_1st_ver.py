import cv2
import numpy as np
import pyrealsense2 as rs
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

# Запуск трансляции с камеры
# cameraCapture = cv2.VideoCapture(0)
# cv2.namedWindow('webcam')

# Создание конвейера для RealSense
pipeline = rs.pipeline()
config = rs.config()

# Настройка конфигурации конвейера
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Запуск конвейера
pipeline.start(config)

try:
    while True:
        # success, frame = cameraCapture.read()

        # Получение кадра с RealSense камеры
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())

        # Преобразование в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Медианное размытие для уменьшение шума
        img = cv2.medianBlur(gray, 37)
        # Обнаружение кругов
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=120, param2=40)

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
                    print("знак 3.1. Въезд запрещен.")
                    z31_stok = cv2.imread('z31.png')
                    z31 = cv2.resize(z31_stok, (500, 500))
                    WindowNameZ31 = '3.1'
                    cv2.imshow(WindowNameZ31, z31)
                    cv2.moveWindow(WindowNameZ31, 0, 0)
                    cv2.waitKey(1000)
                    cv2.destroyWindow(WindowNameZ31)

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
                            print("знак 4.1.3. Движение налево.")
                            z413_stok = cv2.imread('z413.png')
                            height, width, _ = z413_stok.shape
                            z413 = cv2.resize(z413_stok, (width//2, height//2))
                            WindowNameZ413 = '4.1.3'
                            cv2.imshow(WindowNameZ413, z413)
                            cv2.moveWindow(WindowNameZ413, 0, 0)
                            cv2.waitKey(1000)
                            cv2.destroyWindow(WindowNameZ413)
                        else:
                            print("знак 4.1.2. Движение направо.")
                            z412_stok = cv2.imread('z412.png')
                            height, width, _ = z412_stok.shape
                            z412 = cv2.resize(z412_stok, (width//2, height//2))
                            WindowNameZ412 = '4.1.2'
                            cv2.imshow(WindowNameZ412, z412)
                            cv2.moveWindow(WindowNameZ412, 0, 0)
                            cv2.waitKey(1000)
                            cv2.destroyWindow(WindowNameZ412)
                    else:
                        if sum(zone_1_color) > sum(zone_0_color) and sum(zone_1_color) > sum(zone_2_color):
                            print("знак 4.1.1. Движение прямо.")
                            z411_stok = cv2.imread('z411.png')
                            height, width, _ = z411_stok.shape
                            z411 = cv2.resize(z411_stok, (width//2, height//2))
                            WindowNameZ411 = '4.1.1'
                            cv2.imshow(WindowNameZ411, z411)
                            cv2.moveWindow(WindowNameZ411, 0, 0)
                            cv2.waitKey(1000)
                            cv2.destroyWindow(WindowNameZ411)
                        elif sum(zone_0_color) > sum(zone_2_color):
                            print("знак 4.1.5. Движение прямо или налево")
                            z415_stok = cv2.imread('z415.png')
                            height, width, _ = z415_stok.shape
                            z415 = cv2.resize(z415_stok, (width//2, height//2))
                            WindowNameZ415 = '4.1.5'
                            cv2.imshow(WindowNameZ415, z415)
                            cv2.moveWindow(WindowNameZ415, 0, 0)
                            cv2.waitKey(1000)
                            cv2.destroyWindow(WindowNameZ415)
                        else:
                            print("знак 4.1.4. Движение прямо или направо.")
                            z414 = cv2.imread('z414.png')
                            height, width, _ = z414.shape
                            z414 = cv2.resize(z414, (width//2, height//2))
                            WindowNameZ414 = '4.1.4'
                            cv2.imshow(WindowNameZ414, z414)
                            cv2.moveWindow(WindowNameZ414, 0, 0)
                            cv2.waitKey(1000)
                            cv2.destroyWindow(WindowNameZ414)
                # Если программа не может распознать знак, то она обязательно об этом сообщит
                else:
                    print("не распознано")
            # Этот цикл демонстирует распозанные круги, обнаруженные с помощью HoughCircles на рамке
            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
        # # Демонастрация изображения на экран с веб-камеры и остановка демонстации на клавишу esc
        # cv2.imshow('webcam', frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break
        # Демонстрация изображения на экран с RealSense камеры и остановка демонстации на клавишу esc
        cv2.imshow('RealSense', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


# cv2.destroyAllWindows()
# cameraCapture.release()

finally:
    # Остановка конвейера и освобождение ресурсов
    pipeline.stop()
    cv2.destroyAllWindows()
