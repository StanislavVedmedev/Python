import os
import cv2
import numpy as np
import math
import csv
import time
import uuid
from multiprocessing.pool import Pool


def main():
    directory = 'images'

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
    headers = ['Path', 'Area', 'line_a', 'line_b', 'line_c', 'line_d', 'first_angle', 'second_angle',
               'third_angle', 'fourth_angle', 's_dug1', 's_dug2', 's_dug3', 's_dug4', 'zol_h', 'zol_w',
               'height_seed', 'width_seed']
    with open('SunFlowersForma1.csv', 'w', newline='') as f:
        write = csv.writer(f, delimiter=';')

        write.writerow(headers)
        write.writerows(results)
    # -----------------------------------------


def process_image(path):
    print(f'*** Изображение: {path} ***')
    directory, filename = os.path.split(path)
    hsv_min = np.array((28, 148, 12), np.uint8)  # Минимальные значения rgb формата
    hsv_max = np.array((165, 255, 178), np.uint8)  # Максимальные rgb значения формата

    image = cv2.imread(path)
    thresh = cv2.inRange(image, hsv_min, hsv_max)

    temp_name = f'{uuid.uuid4()}.jpg'
    cv2.imwrite(temp_name, thresh)
    temp_image = cv2.imread(temp_name)
    os.remove(temp_name)

    r, g, b = cv2.split(temp_image)
    bit_matrix_r = (r > 250)
    bit_matrix_g = (g > 250)
    bit_matrix_b = (b > 250)
    bit_matrix = np.multiply(np.multiply(bit_matrix_r, bit_matrix_g), bit_matrix_b)
    numb_matrix = bit_matrix.astype(int)
    numb_matrix = np.array(numb_matrix)
    white_pixels_sum = 0
    for i in range(len(numb_matrix)):
        for j in range(len(numb_matrix[i])):
            white_pixels_sum += numb_matrix[i][j]

    print(f'*** Площадь растения: {white_pixels_sum} ***')

    max_row_coordinates, max_col_coordinates = find_coordinates(numb_matrix)

    # print("Координаты крайних элементов, равных 1, по строке с максимальной суммой:", max_row_coordinates)
    # print("Координаты крайних элементов, равных 1, по столбцу с максимальной суммой:", max_col_coordinates)
    min_r = min(max_row_coordinates)
    max_r = max(max_row_coordinates)
    max_c = max(max_col_coordinates)
    min_c = min(max_col_coordinates)
    x_min_r_coordinate = min_r[0]
    y_min_r_coordinate = min_r[1]
    x_max_r_coordinate = max_r[0]
    y_max_r_coordinate = max_r[1]
    x_min_c_coordinate = min_c[0]
    y_min_c_coordinate = min_c[1]
    x_max_c_coordinate = max_c[0]
    y_max_c_coordinate = max_c[1]
    l_1 = [[x_min_r_coordinate, y_min_r_coordinate],
           [x_max_r_coordinate, y_max_r_coordinate]]  # Координаты начала и конца линии по длинне семечки
    l_2 = [[x_min_c_coordinate, y_min_c_coordinate],
           [x_max_c_coordinate, y_max_c_coordinate]]  # Координаты начала и конца линии по ширине семечки
    xd = (l_1[0][0] - l_1[1][0], l_2[0][0] - l_2[1][0])
    yd = (l_1[0][1] - l_1[1][1], l_2[0][1] - l_2[1][1])
    div = xd[0] * yd[1] - xd[1] * yd[0]

    if div == 0:
        raise Exception('lines do not intersect')

    d = (l_1[0][0] * l_1[1][1] - l_1[0][1] * l_1[1][0], l_2[0][0] * l_2[1][1] - l_2[0][1] * l_2[1][0])
    x = (d[0] * xd[1] - d[1] * xd[0]) / div  # Кооордината пересечения по x
    y = (d[0] * yd[1] - d[1] * yd[0]) / div  # Кооордината пересечения по y

    line_a = y - y_min_r_coordinate
    line_b = y_max_r_coordinate - y
    line_c = x - x_min_c_coordinate
    line_d = x_max_c_coordinate - x
    rs1 = line_a * line_c / 2  # Площа 1 трикутника
    rs2 = line_c * line_b / 2  # Площа 2 трикутника
    rs3 = line_b * line_d / 2  # Площа 3 трикутника
    rs4 = line_a * line_d / 2  # Площа 4 трикутника
    rmb = rs1 + rs2 + rs3 + rs4

    r_proportion1 = rmb / rs1
    r_proportion2 = rmb / rs2
    r_proportion3 = rmb / rs3
    r_proportion4 = rmb / rs4

    square1 = white_pixels_sum / r_proportion1
    square2 = white_pixels_sum / r_proportion2
    square3 = white_pixels_sum / r_proportion3
    square4 = white_pixels_sum / r_proportion4
    s_dug1 = square1 - rs1
    s_dug2 = square2 - rs2
    s_dug3 = square3 - rs3
    s_dug4 = square4 - rs4
    hip1 = math.sqrt(line_a ** 2 + line_c ** 2)
    hip2 = math.sqrt(line_c ** 2 + line_b ** 2)
    hip3 = math.sqrt(line_c ** 2 + line_d ** 2)
    hip4 = math.sqrt(line_a ** 2 + line_d ** 2)

    beta1 = math.asin(line_c / hip1) * (180 / math.pi)
    beta2 = math.asin(line_c / hip2) * (180 / math.pi)
    beta3 = math.asin(line_d / hip3) * (180 / math.pi)
    beta4 = math.asin(line_d / hip4) * (180 / math.pi)

    alpha1 = 90 - beta1  # Первая половина 2 угла
    alpha2 = 90 - beta2  # Первал половина 4 угла
    alpha3 = 90 - beta3  # Вторая половина 4 угла
    alpha4 = 90 - beta4  # Вторая половина 2 угла

    first_angle = beta1 + beta4  # Первый угол
    second_angle = alpha1 + alpha2  # Второй укол
    third_angle = beta2 + beta3  # Третий угол
    fourth_angle = alpha3 + alpha4  # Четвёртый угол
    zol_h = (line_c + line_d) / line_c  # Соотношение участков ширин
    zol_w = (line_a + line_b) / line_a  # Соотношение участков длин
    width_seed = line_a + line_b  # Ширина семечки
    height_seed = line_c + line_d  # Длинна семечки

    return [filename, white_pixels_sum, line_a, line_b, line_c, line_d, first_angle, second_angle, third_angle,
            fourth_angle, s_dug1, s_dug2, s_dug3, s_dug4, zol_h, zol_w, height_seed, width_seed]


def find_coordinates(numb_matrix):
    max_row_index = find_max_row(numb_matrix)
    max_col_index = find_max_column(numb_matrix)

    max_row = numb_matrix[max_row_index]
    max_col = [numb_matrix[i][max_col_index] for i in range(len(numb_matrix))]

    max_row_ones = [i for i, val in enumerate(max_row) if val == 1]
    max_col_ones = [i for i, val in enumerate(max_col) if val == 1]

    return [(max_row_index, col) for col in max_row_ones], [(row, max_col_index) for row in max_col_ones]


def find_max_row(numb_matrix):
    max_sum = 0
    max_row_index = -1
    for i, row in enumerate(numb_matrix):
        row_sum = sum(row)
        if row_sum > max_sum:
            max_sum = row_sum
            max_row_index = i
    return max_row_index

    # Press the green button in the gutter to run the script.


def find_max_column(numb_matrix):
    max_sum = 0
    max_col_index = -1
    for j in range(len(numb_matrix[0])):
        col_sum = sum(numb_matrix[i][j] for i in range(len(numb_matrix)))
        if col_sum > max_sum:
            max_sum = col_sum
            max_col_index = j
    return max_col_index

    # print("Координаты крайних элементов, равных 1, по строке с максимальной суммой:", max_row_coordinates)
    # print("Координаты крайних элементов, равных 1, по столбцу с максимальной суммой:", max_col_coordinates)


if __name__ == '__main__':
    main()

