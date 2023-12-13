import sys
from PyQt5.QtCore import Qt 
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Classifier App')

        # Кнопка для выбора изображения
        self.choose_button = QPushButton('Выбрать изображение', self)
        self.choose_button.clicked.connect(self.choose_image)

        # Метка для отображения результата
        self.result_label = QLabel('Результат: ', self)

        # Метка для отображения выбранного изображения
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Основной макет
        layout = QVBoxLayout()
        layout.addWidget(self.choose_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        # Путь к файлу модели
        self.model_path = r"C:\Users\Daniil\Desktop\name.h5" 
        # Загрузка модели
        self.model = load_model(self.model_path)

    def choose_image(self):
        # Диалоговое окно для выбора файла изображения
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Выбрать изображение', '', 'Image files (*.png *.jpg *.jpeg)')

        if file_path:
            # Загрузка и предобработка изображения
            prepared_image = self.load_and_prepare_image(file_path)

            # Получение предсказания от модели
            predictions = self.model.predict(prepared_image)
            predicted_class_index = np.argmax(predictions)
            class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']
            predicted_class = class_labels[predicted_class_index]

            # Обновление метки с результатом
            self.result_label.setText(f'Предсказанный класс: {predicted_class}')

            # Отображение выбранного изображения
            self.display_image(file_path)

    def load_and_prepare_image(self, img_path, img_size=(200, 200)):
        img = image.load_img(img_path, target_size=img_size, color_mode="grayscale")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def display_image(self, img_path):
        # Отображение выбранного изображения
        pixmap = QPixmap(img_path)
        pixmap = pixmap.scaledToHeight(200)
        self.image_label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageClassifierApp()
    ex.show()
    sys.exit(app.exec_())
