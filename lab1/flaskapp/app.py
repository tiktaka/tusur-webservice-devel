import os
from flask import Flask, render_template, request
from flask_wtf import FlaskForm, RecaptchaField
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
# Загружаем ключи из переменных окружения для reCAPTCHA
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['RECAPTCHA_PUBLIC_KEY'] = os.getenv('RECAPTCHA_PUBLIC_KEY')
app.config['RECAPTCHA_PRIVATE_KEY'] = os.getenv('RECAPTCHA_PRIVATE_KEY')

# Применение функции к изображению
def apply_function(image, period, axis, func_type='sin'):
    img_array = np.array(image).astype(np.float32)  # Преобразуем изображение в массив
    height, width = img_array.shape[:2] # Берём только высоту/ширину

    # Выбор оси
    if axis == 'horizontal':
        axis_values = np.linspace(0, 2 * np.pi * width / period, width)
        axis_array = np.tile(axis_values, (height, 1))
    else:
        axis_values = np.linspace(0, 2 * np.pi * height / period, height)
        axis_array = np.tile(axis_values, (width, 1)).T

    # Применяем выбранную функцию sin/cos
    if func_type == 'sin':
        multiplier = np.sin(axis_array)
    else:
        multiplier = np.cos(axis_array)

    # Применяем функцию к R,G и B каналам
    for i in range(3):
        img_array[..., i] = img_array[..., i] * (multiplier + 1) / 2

    # Проверяем соблюдение диапазона значений 0-255 в массиве и возвращаем результат пребразования
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

# Функция для построения графика распределения цветов
def color_distribution_graph(image):
    img_array = np.array(image)
    
    plt.figure(figsize=(10, 6)) # Размечаем график
    labels=['red', 'green', 'blue'] # Обозначаем метки для легенды на графике

    # Для каждого канала
    for i, color in enumerate(labels):
        pixel_values = img_array[..., i].ravel()  # делаем одномерный массив
        
        # Подсчитываем распределение используя гистограмму
        counts, bins = np.histogram(pixel_values, bins=256, range=(0, 256))
        
        # берём только середины интервалов из гистограммы, так как для графика нужны конкретные числа, а не интервалы
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # добавляем кривую на график
        plt.plot(bin_centers, counts, color=color, linewidth=2)

    plt.legend(labels) # добавляем метки цветов на легенду
    # Даём названия графику, осям и рисуем сетку
    plt.title('Распределение цветов')
    plt.xlabel('Значение цвета')
    plt.ylabel('Кол-во')
    plt.grid(True, alpha=0.3)
    
    # Пишем массив пикселей в буфер в формате png и ставим указатель в начало
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    # возвращаем буфер с изображением внутри
    return img_buffer

# Функция для преобразования изобрадения в base64 для вставки в HTML
def image_to_base64(image_buffer):
    image_buffer.seek(0)
    return base64.b64encode(image_buffer.read()).decode('utf-8')

# Форма с капчей
class ImageProcessingForm(FlaskForm):
    recaptcha = RecaptchaField()
    pass

# Главная страница с формой
@app.route('/')
def index():
    form = ImageProcessingForm() # добавляем форму с капчей в рендер страницы
    return render_template('index.html', form=form)

# Страница с результатом преобразования
@app.route('/process', methods=['POST'])
def process_image():
    # СОздаём форму, чтобы проверить капчу, если она не пройдена - возвращаем ошибку
    form = ImageProcessingForm()
    if not form.recaptcha.validate(form):
        return "Ошибка: не пройдена проверка reCAPTCHA. Попробуйте снова.", 400

    # Получаем данные с формы
    image = Image.open(request.files['image'])
    period = float(request.form['period'])
    axis = request.form['axis']
    func_type = request.form['function']

    # Преобразуем изображение по указанным в форме параметрам
    processed_image = apply_function(image, period, axis, func_type)

    # Создаем буфер для сохранения изображений до/после
    original_buffer = io.BytesIO()
    processed_buffer = io.BytesIO()

    # Сохраняем изображения в буфер
    image.save(original_buffer, format='PNG')
    processed_image.save(processed_buffer, format='PNG')

    # Преобразуем изображения в base64
    original_base64 = image_to_base64(original_buffer)
    processed_base64 = image_to_base64(processed_buffer)

    # Генерируем графики
    original_plot = color_distribution_graph(image)
    processed_plot = color_distribution_graph(processed_image)

    # Графики тоже преобразуем в base64
    original_plot_base64 = image_to_base64(original_plot)
    processed_plot_base64 = image_to_base64(processed_plot)

    # Рендерим страницу с результатами работы
    return render_template('result.html', 
                           original_image=original_base64,
                           processed_image=processed_base64,
                           original_plot=original_plot_base64,
                           processed_plot=processed_plot_base64)

if __name__ == '__main__':
    app.run(debug=True)