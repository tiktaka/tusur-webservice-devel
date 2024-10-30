from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Функция для применения периодической функции к изображению
def apply_periodic_function(image, period, axis, func_type='sin'):
    img_array = np.array(image).astype(np.float32)  # Преобразуем изображение в массив
    height, width = img_array.shape[:2]

    # Выбор оси для аргумента функции
    if axis == 'horizontal':
        axis_values = np.linspace(0, 2 * np.pi * width / period, width)
        axis_array = np.tile(axis_values, (height, 1))
    else:
        axis_values = np.linspace(0, 2 * np.pi * height / period, height)
        axis_array = np.tile(axis_values, (width, 1)).T

    # Применяем выбранную функцию (sin или cos)
    if func_type == 'sin':
        multiplier = np.sin(axis_array)
    else:
        multiplier = np.cos(axis_array)

    # Применяем функцию ко всем каналам изображения (для RGB)
    for i in range(3):  # Для R, G, B каналов
        img_array[..., i] = img_array[..., i] * (multiplier + 1) / 2

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

# Функция для построения графика распределения цветов
def plot_color_distribution(image):
    img_array = np.array(image)
    colors = ('r', 'g', 'b')
    plt.figure(figsize=(6, 4))
    
    for i, color in enumerate(colors):
        plt.hist(img_array[..., i].ravel(), bins=256, color=color, alpha=0.5)
    
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Color Distribution')
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    return img_buffer

# Маршрут для отображения главной страницы
@app.route('/')
def index():
    return render_template('index.html')

import base64

def image_to_base64(image_buffer):
    image_buffer.seek(0)
    return base64.b64encode(image_buffer.read()).decode('utf-8')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image = Image.open(request.files['image'])
    period = float(request.form['period'])
    axis = request.form['axis']
    func_type = request.form['function']

    # Преобразуем изображение
    processed_image = apply_periodic_function(image, period, axis, func_type)

    # Создаем буфер для сохранения изображений
    original_buffer = io.BytesIO()
    processed_buffer = io.BytesIO()

    # Сохраняем изображения в буфер
    image.save(original_buffer, format='PNG')
    processed_image.save(processed_buffer, format='PNG')

    # Преобразуем изображения в base64
    original_base64 = image_to_base64(original_buffer)
    processed_base64 = image_to_base64(processed_buffer)

    # Генерируем графики
    original_plot = plot_color_distribution(image)
    processed_plot = plot_color_distribution(processed_image)

    original_plot_base64 = image_to_base64(original_plot)
    processed_plot_base64 = image_to_base64(processed_plot)

    return render_template('result.html', 
                           original_image=original_base64,
                           processed_image=processed_base64,
                           original_plot=original_plot_base64,
                           processed_plot=processed_plot_base64)


if __name__ == '__main__':
    app.run(debug=True)
