from datetime import datetime
from os import getenv
from flask import Flask, render_template, request
from flask_wtf import FlaskForm, RecaptchaField
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode

app = Flask(__name__)
# Загружаем ключи из переменных окружения для reCAPTCHA
app.config['SECRET_KEY'] = getenv('SECRET_KEY')
app.config['RECAPTCHA_PUBLIC_KEY'] = getenv('RECAPTCHA_PUBLIC_KEY')
app.config['RECAPTCHA_PRIVATE_KEY'] = getenv('RECAPTCHA_PRIVATE_KEY')

# Применение функции к изображению
def apply_function(image, period, axis, func_type='sin'):
    img_array = np.array(image).astype(np.float32)  # Преобразуем изображение в массив
    height, width = img_array.shape[:2] # Берём срезом высоту/ширину

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
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    # возвращаем буфер с изображением внутри
    return img_buffer

# Добавление даты и времени на изображение (в правый нижний угол)
def draw_datetime(image):
    draw = ImageDraw.Draw(image)
    # получаем текущую дату и время и преобразуем в строку с понятным форматом
    current_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    # Устанавливаем размер шрифта, чтобы дата и время занимали 5% ширины
    font = ImageFont.load_default(size=image.width * 0.05)
    # Получаем границы текста относительно 0 позиции на изображении (уже с учётом размера шрифта)
    sizes = draw.textbbox((0, 0), current_datetime, font=font)
    # Из границ получаем размеры текста с датой в временем
    datetime_height = sizes[3] - sizes[1]
    datetime_width = sizes[2] - sizes[0]
    # Вычитаем из размеров изображения размеры текста + вычитаем фиксированный отступ
    x = image.width - datetime_width - 20
    y = image.height - datetime_height - 20
    # Добавляем строку с датой и временем  на картинку
    draw.text((x, y), current_datetime, fill=(255, 0, 0), font=font)
    
# Функция для преобразования изобрадения в base64 для вставки в HTML
def image_to_base64(image_buffer):
    image_buffer.seek(0)
    return b64encode(image_buffer.read()).decode('utf-8')

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

    # Генерируем графики
    original_plot = color_distribution_graph(image)
    processed_plot = color_distribution_graph(processed_image)

    # Графики тоже преобразуем в base64
    original_plot_base64 = image_to_base64(original_plot)
    processed_plot_base64 = image_to_base64(processed_plot)

    # Если чекбокс установлен, то генерируем на изображениии дату и время
    if "draw_datetime" in request.form:
        draw_datetime(processed_image)

    # Создаем буфер для сохранения изображений до/после
    original_buffer = BytesIO()
    processed_buffer = BytesIO()

    # Сохраняем изображения в буфер
    image.save(original_buffer, format='PNG')
    processed_image.save(processed_buffer, format='PNG')

    # Преобразуем изображения в base64
    original_base64 = image_to_base64(original_buffer)
    processed_base64 = image_to_base64(processed_buffer)

    # Рендерим страницу с результатами работы
    return render_template('result.html', 
                           original_image=original_base64,
                           processed_image=processed_base64,
                           original_plot=original_plot_base64,
                           processed_plot=processed_plot_base64)

if __name__ == '__main__':
    app.run(debug=True)