import os
import cv2
import numpy as np
import csv
import time
import imutils
from multiprocessing.pool import Pool


def main():
    directory = 'seeds2023-234-239'

    tp1 = time.time()

    images = []
    for filename in os.listdir(directory):
        file_path = os.path.abspath(os.path.join(directory, filename))
        if os.path.isfile(file_path):
            images.append(file_path)

    p = Pool(processes=4)
    results = p.map(process_image, images)
    p.close()
    p.join()

    tp2 = time.time()
    print(f'*** Общее время: {tp2 - tp1} ***')

    # ----------------------------------------
    # Строки 28 - 33. Сохранение результатов в файл csv (разделитель ";")
    headers = ['Название', 'Ширина', 'Высота', 'Цвет', 'Красный', 'Зеленый', 'Синий', 'Площадь']
    with open('seeds2023-234-239size.csv', 'w', newline='') as f:
        write = csv.writer(f, delimiter=';')

        write.writerow(headers)
        write.writerows(results)
        #for result in results:
            #write.writerow(result)
    # -----------------------------------------


# "process_image" - основная функция для рассчета высоты/ширины отдельной семечки. В качестве параметра принимает "абсолютный" путь к фотографии семечки
def process_image(path):
    print(f'*** Изображение: {path} ***')
    # ----------------------------------------
    # Строки 41 - 47. Преобразование фотографии в матрицу нулей и едениц
    image = cv2.imread(path)
    r, g, b = cv2.split(image)
    bit_matrix_r = (r < 115)
    bit_matrix_g = (g < 115)
    bit_matrix_b = (b < 115)
    bit_matrix = np.multiply(np.multiply(bit_matrix_r, bit_matrix_g), bit_matrix_b)
    numb_matrix = bit_matrix.astype(int)
    # ----------------------------------------

    square_matrix = get_square_matrix(numb_matrix)  # Приведение матрицы к "квадратному" виду посредством вставки нулей (чтобы число строк равнялось числу столбцов).

    primary_diagonal_items = []  # массив, в котором будут храниться диагонали, параллельные главной (если точнее, то хранится сумма едениц для каждой диагонали, что параллельна главной)
    secondary_diagonal_items = []  # массив, в котором будут храниться диагонали, параллельные побочной (если точнее, то хранится сумма едениц для каждой диагонали, что параллельна побочной)
    for i in range(0, len(square_matrix)):  # Цикл от нуля до "размера матрицы"

        primary_diagonal_sum = sum_primary_diagonal(square_matrix[len(square_matrix) - (i + 1):len(square_matrix), 0:i + 1])  # С нижнего левого угла основной матрицы берется "подматрица" (квадратик) и для нее рассчитывается сумма по главной диагонали
        primary_diagonal_items.append(primary_diagonal_sum)  # Сохранение суммы по главной диагонали "подматрицы" для последующего определения максимальной

        primary_diagonal_sum = sum_primary_diagonal(square_matrix[0:i + 1, len(square_matrix) - (i + 1):len(square_matrix)])  # С верхнего правого угла основной матрицы берется "подматрица" (квадратик) и для нее рассчитывается сумма по главной диагонали
        primary_diagonal_items.append(primary_diagonal_sum)  # Сохранение суммы по главной диагонали "подматрицы" для последующего определения максимальной

        secondary_diagonal_sum = sum_secondary_diagonal(square_matrix[0:i + 1, 0:i + 1])  # С верхнего левого угла основной матрицы берется "подматрица" (квадратик) и для нее рассчитывается сумма по побочной диагонали
        secondary_diagonal_items.append(secondary_diagonal_sum)  # Сохранение суммы по побочной диагонали "подматрицы" для последующего определения максимальной

        secondary_diagonal_sum = sum_secondary_diagonal(square_matrix[len(square_matrix) - (i + 1):len(square_matrix), len(square_matrix) - (i + 1):len(square_matrix)])  # С нижнего правого угла основной матрицы берется "подматрица" (квадратик) и для нее рассчитывается сумма по главной диагонали
        secondary_diagonal_items.append(secondary_diagonal_sum)  # Сохранение суммы по побочной диагонали "подматрицы" для последующего определения максимальной

    width = max(secondary_diagonal_items)  # рассчет максимальной диагонали, параллельной побочной (максимальной суммы едениц)
    height = max(primary_diagonal_items)  # рассчет максимальной диагонали, параллельной главной (максимальной суммы едениц)

    red, green, blue = get_image_color(image)
    summa = 0
    for i in range(len(square_matrix)):
        for j in range(len(square_matrix[i])):
            summa += square_matrix[i][j]

    return [os.path.basename(path), width, height, 'rgb', red, green, blue, summa]


# "get_square_matrix" - функция для приведения матрицы в квадратный вид (недостающие элементы заполняются нулями)
def get_square_matrix(matrix, val=0):
    (a, b) = matrix.shape

    if a > b:
        padding = ((0, 0), (0, a - b))
    else:
        padding = ((0, b - a), (0, 0))

    return np.pad(matrix, padding, mode='constant', constant_values=val)


# "sum_primary_diagonal" - функция для рассчета суммы чисел на главной диагонали матрицы (с верхнего левого угла до нижнего правого)
def sum_primary_diagonal(matrix):
    diagonal_sum = 0

    for i in range(0, len(matrix)):
        diagonal_sum += matrix[i][i]

    return diagonal_sum


# "sum_secondary_diagonal" - функция для рассчета суммы чисел на побочной диагонали матрицы (с верхнего правого угла
# до нижнего левого)
def sum_secondary_diagonal(matrix):
    diagonal_sum = 0

    for i in range(0, len(matrix)):
        diagonal_sum += matrix[i][len(matrix) - (i + 1)]

    return diagonal_sum


def get_image_color(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours_len = [len(c) for c in contours]
        max_contour = contours[contours_len.index(max(contours_len))]

        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [max_contour], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]
        blue_mean = round(mean[0])
        green_mean = round(mean[1])
        red_mean = round(mean[2])

        return red_mean, green_mean, blue_mean
    except:
        return '-'


if __name__ == '__main__':
    main()