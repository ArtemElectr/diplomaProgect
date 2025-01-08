import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import numpy as np

import PIL
import tensorflow as tf

import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_dir = ''#r"F:\urban_diploma\pythonProject\archive\seg_train\seg_train"
test_data_dir = ''#r'F:\urban_diploma\pythonProject\archive\seg_train\seg_train'
data_dir = pathlib.Path(data_dir)
batch_size = 32
img_height = 150
img_width = 150
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

image_count = len(list(data_dir.glob('*/*.jpg')))  # Функция glob.glob () используется для поиска всех файлов,
#print('1 ',image_count)
print(data_dir)

buildings = list(data_dir.glob('buildings/*'))

#print(buildings)
PIL.Image.open(str(buildings[0]))

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    labels='inferred',  #  либо «inferred» (метки генерируются из структуры директории), либо NULL (без меток), либо
    # список или кортеж целочисленных меток такого же размера, как количество файлов изображений в директории. Метки
    # должны быть отсортированы в алфавитном порядке путей к файлам изображений.
    validation_split=0.2, # Необязательный параметр с плавающей точкой от 0 до 1, доля данных,резервируемых для проверки
    label_mode='int',
    subset="training",
    class_names=class_names,  # class_names — только если метки «inferred». Это явный список имён классов (должен
    # соответствовать именам поддиректорий). Используется для контроля порядка классов (иначе используется алфавитный
    # порядок)
    seed=123,   # необязательное случайное семя для перетасовки и преобразований.
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True  # Перетасовывать ли данные. По умолчанию — True. Если установить False, сортирует данные в алфавитном
                    # порядке
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  directory=test_data_dir,
  labels='inferred',
  label_mode='int',
  validation_split=0.2,
  subset="validation",
  class_names=class_names,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)
class_names = train_ds.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

    #plt.show()

print('TRAIN: ', train_ds)
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)  # Это пакет из 32 изображений размером 180x180x3 (последний размер относится к цветовым каналам RGB).
    print(labels_batch.shape) # метки для изобр
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Обучите модель

epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Визуализируйте результаты тренировок. Создайте графики потерь и точности на обучающих и проверочных наборах:
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
print(model.get_metrics_result())
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

sunflower_url = r"F:\urban_diploma\pythonProject\archive\seg_pred\seg_pred\3.jpg"

img = tf.keras.utils.load_img(
    sunflower_url, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)