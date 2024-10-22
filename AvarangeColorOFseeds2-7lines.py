import os
import csv
import numpy as np
import pandas as pd
from PIL import Image

IMAGES_DIRECTORY = '2023-356'
STATISTICS_DIRECTORY = 'statistics'
IMAGE_SIZE = 1200
LINES_COUNT = 8


def main():
    results = []
    for filename in os.listdir(IMAGES_DIRECTORY):  # В цикле проходим по каждому файлу из папки images
        file_path = os.path.abspath(os.path.join(IMAGES_DIRECTORY, filename))
        if os.path.isfile(file_path):
            results.append(process_image(file_path))

    headers = get_headers()
    with open('2023-356lines.csv', 'w', newline='') as f:
        write = csv.writer(f, delimiter=';')

        write.writerow(headers)
        write.writerows(results)


def get_headers():
    headers = ['Name']
    for i in range(1, LINES_COUNT-1):
        headers += [f'red_avg{i + 1}', f'green_avg{i + 1}', f'blue_avg{i + 1}']

    return headers


def process_image(path):
    image = Image.open(path)  # Открываем изображение
    new_size = (IMAGE_SIZE, IMAGE_SIZE)
    resized_image = image.resize(new_size)  # Задаем новые размеры изображения
    # resized_image.show()
    filename = os.path.basename(path)  # Получаем название файла
    new_filename = 'resized_' + filename  # Задаем новое название файла
    resized_image.save(new_filename)  # Сохраняем изображение с новыми размерами
    pix = resized_image.load()  # Выгружаем значения пикселей
    os.remove(new_filename)  # Удаляем изображение с новыми размерами

    return process_image_pix(filename, pix)


def process_image_pix(filename, pix):
    if not os.path.exists(STATISTICS_DIRECTORY):
        os.makedirs(STATISTICS_DIRECTORY)  # Создаем папку statistics если ее не существует

    csv_filename = os.path.join(STATISTICS_DIRECTORY, os.path.splitext(filename)[0] + '.csv')  # Задаем название csv файла
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        headers = ['x', 'y', 'r', 'g', 'b', 'r_razn', 'g_razn', 'b_razn']  # Заголовки столбцов
        writer.writerow(headers)

        for i in range(1, IMAGE_SIZE):
            r, g, b = pix[i, i]  # Определяем значения пикселей по главной диагонали
            rrazn, grazn, brazn = abs(abs(np.array(pix[i, i]) - np.array(pix[i - 1, i - 1])))  # Определяю разницу по модулю последующего и предыдущего значений
            results = [i, i, r, g, b, rrazn, grazn, brazn]
            writer.writerow(results)

    statistics = pd.read_csv(csv_filename, delimiter=',')
    statistics.sort_values(by="r_razn", ascending=False)
    newdf = statistics.nlargest(LINES_COUNT, "r_razn")

    return [filename] + process_statistics(newdf, statistics)


def process_statistics(newdf, statistics):
    max_values = []
    for i in range(0, LINES_COUNT - 1):
        max_val = max(x for x in newdf['x'] if x not in max_values)
        max_values.append(max_val)

    max_values.reverse()

    filtered_statistics = [statistics[statistics.x < max_values[0]]]
    for i in range(1, len(max_values)):
        filtered_statistics.append(statistics[(statistics['x'] < max_values[i]) & (statistics['x'] >= max_values[i - 1])])
    filtered_statistics.append(statistics[statistics.x >= max_values[-1]])

    results = []
    for i in range(1, LINES_COUNT-1):
        red_avg = int(sum(filtered_statistics[i]['r']) / len(filtered_statistics[i]))
        green_avg = int(sum(filtered_statistics[i]['g']) / len(filtered_statistics[i]))
        blue_avg = int(sum(filtered_statistics[i]['b']) / len(filtered_statistics[i]))
        results += [red_avg, green_avg, blue_avg]

    return results


if __name__ == '__main__':
    main()
