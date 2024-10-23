import os
import cv2
import numpy as np
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
    headers = ['Название', 'Площадь растения']
    with open('FourSunflowers.csv', 'w', newline='') as f:
        write = csv.writer(f, delimiter=';')

        write.writerow(headers)
        write.writerows(results)
    # -----------------------------------------


def process_image(path):
    print(f'*** Изображение: {path} ***')

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

    white_pixels_sum = 0
    for i in range(len(numb_matrix)):
        for j in range(len(numb_matrix[i])):
            white_pixels_sum += numb_matrix[i][j]

    print(f'*** Площадь растения: {white_pixels_sum} ***')

    return [os.path.basename(path), white_pixels_sum]


if __name__ == '__main__':
    main()
