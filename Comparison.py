import os
import pickle
import cv2
import numpy as np
import faiss
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

# Загрузка модели EfficientNetB7
model = EfficientNetB7(weights='imagenet', include_top=False, pooling='avg')

# Папка с сохраненными признаками
features_folder = "D:/find_way/features_241"
scrin_features = []
scrin_filenames = []

# Загрузка всех признаков из файлов
for feature_file in os.listdir(features_folder):
    if not feature_file.endswith('.pkl'):
        continue
    with open(os.path.join(features_folder, feature_file), 'rb') as f:
        features = pickle.load(f)
        scrin_features.extend(features)
        scrin_filenames.extend([feature_file] * len(features))  # Сохраняем название файла для всех 360 поворотов

# Преобразование списка в numpy массив
scrin_features = np.array(scrin_features).astype('float32')

# Создание индекса FAISS
d = scrin_features.shape[1]  # Размерность признаков
index = faiss.IndexFlatL2(d)
index.add(scrin_features)

# Пути к папке с изображениями из path_not
logs_folder = "D:/find_way/path_not"

# Предобработка и поиск похожих изображений для изображений из path_not
for filename in os.listdir(logs_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.PNG')):
        continue
    image_path = os.path.join(logs_folder, filename)
    try:
        # Предобработка изображения из path_not
        img = preprocess_and_transform_image(image_path)
        img_expanded = np.expand_dims(img, axis=0)
        feature = model.predict(img_expanded)
        feature = feature.astype('float32')

        # Определяем начальный размер топа для поиска
        top_k = 5
        max_top_k = 100  # Максимальное количество совпадений для поиска
        found_unique = False

        # Поиск совпадений с увеличением топа
        while top_k <= max_top_k:
            D, I = index.search(feature, top_k)

            # Фильтрация для уникальных названий
            unique_results = []
            unique_names = set()

            for distance, index_id in zip(D[0], I[0]):
                similar_filename = scrin_filenames[index_id]
                # Извлекаем оригинальное имя файла без "повернуто на n градусов"
                original_filename = similar_filename.split('_features')[0]
                if original_filename not in unique_names:
                    unique_names.add(original_filename)
                    unique_results.append((original_filename, distance))
                if len(unique_results) == 10:
                    found_unique = True
                    break

            # Если нашли 5 уникальных изображений, прекращаем поиск
            if found_unique:
                break

            # Увеличиваем размер топа и продолжаем поиск
            top_k += 10

        # Выводим результаты (даже если их меньше 5, если top_k достиг 100)
        print(f"\nТоп-5 уникальных похожих изображений для {filename}:")
        for idx, (similar_filename, distance) in enumerate(unique_results, 1):
            print(f"{idx}. {similar_filename} (Расстояние: {distance})")

    except Exception as e:
        print(f"Ошибка при обработке {filename}: {e}")
        continue