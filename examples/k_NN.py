import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# Загрузка данных MNIST
mnist = fetch_openml('mnist_784', version=1)
print(mnist.data.shape)
print(mnist.target.size)
X, y = mnist.data / 255.0, mnist.target.astype(int)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


