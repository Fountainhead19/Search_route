import cv2
import numpy as np
import os
import pickle
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_and_transform_image(image_path):
    # Загрузка изображения в формате BGR
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # Преобразование в формат RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Определение диапазона цвета кривой
    target_color = np.array([50, 140, 3])
    tolerance = 40
    lower_color = np.clip(target_color - tolerance, 0, 255)
    upper_color = np.clip(target_color + tolerance, 0, 255)

    # Создание маски на основе цвета
    mask = cv2.inRange(image_rgb, lower_color, upper_color)

    # Нахождение контуров на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"Контуры не найдены на изображении: {image_path}")

    # Выбор самого большого контура
    contour = max(contours, key=cv2.contourArea)

    # Получение минимального оборачивающего прямоугольника
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype("float32")

    # Вычисление ширины и высоты прямоугольника
    width = int(rect[1][0])
    height = int(rect[1][1])

    # Проверка на нулевые размеры и корректировка
    if width == 0 or height == 0:
        raise ValueError(f"Некорректные размеры прямоугольника на изображении: {image_path}")

    # Определение точек назначения для преобразования с сохранением пропорций
    if width > height:
        result_width = 480
        result_height = int((height / width) * 480)
    else:
        result_height = 480
        result_width = int((width / height) * 480)

    dst_pts = np.array([
        [0, result_height - 1],
        [0, 0],
        [result_width - 1, 0],
        [result_width - 1, result_height - 1]
    ], dtype="float32")

    # Вычисление матрицы перспективного преобразования и применение его
    M = cv2.getPerspectiveTransform(box, dst_pts)
    warped = cv2.warpPerspective(mask, M, (result_width, result_height))

    # Добавление отступов, чтобы сделать изображение 480x480
    delta_w = 480 - result_width
    delta_h = 480 - result_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    warped_padded = cv2.copyMakeBorder(warped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # Конвертация в формат RGB
    warped_padded_rgb = cv2.cvtColor(warped_padded, cv2.COLOR_GRAY2RGB)

    # Изменение размера до 600x600 для EfficientNetB7
    warped_resized = cv2.resize(warped_padded_rgb, (600, 600), interpolation=cv2.INTER_LINEAR)

    # Предобработка для EfficientNetB7
    warped_resized = preprocess_input(warped_resized.astype(np.float32))

    return warped_resized

def rotate_image(image, angle):
    # Получаем центр изображения
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Создаем матрицу поворота
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Поворачиваем изображение
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return rotated

# Загрузка модели EfficientNetB7 без верхнего уровня
model = EfficientNetB7(weights='imagenet', include_top=False, pooling='avg')

# Папка для сохранения признаков
output_folder = "D:/find_way/features_241"
os.makedirs(output_folder, exist_ok=True)

# Пути к папке с изображениями
screenshots_folder = "D:/find_way/scrin_new_good"

# Предобработка и сохранение признаков для всех изображений и их поворотов на 360 градусов
for filename in os.listdir(screenshots_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.PNG')):
        continue
    image_path = os.path.join(screenshots_folder, filename)
    try:
        print(f"Обрабатываем {filename}...")
        features_for_image = []

        # Чтение и предобработка оригинального изображения
        original_img = preprocess_and_transform_image(image_path)

        # Генерация 360 повернутых версий изображения
        for angle in range(360):
            rotated_image = rotate_image(original_img, angle)
            rotated_image = np.expand_dims(rotated_image, axis=0)  # Добавляем измерение для batch
            feature = model.predict(rotated_image)
            features_for_image.append(feature[0])

        # Сохранение признаков в файл с использованием pickle
        with open(os.path.join(output_folder, f"{filename}_features.pkl"), 'wb') as f:
            pickle.dump(features_for_image, f)

    except Exception as e:
        print(f"Ошибка при обработке {filename}: {e}")
        continue