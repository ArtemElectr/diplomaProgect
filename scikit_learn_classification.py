import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import imageio.v3 as iio
from PIL import Image


train_image_dir = 'archive/seg_train/seg_train'
test_image_dir = 'archive/seg_test/seg_test'
categories = os.listdir(train_image_dir)
df = pd.DataFrame()
df_t =  pd.DataFrame()
#print(new_arr)
def set_dataset(df1, image_dir):
    np_arr = np.zeros((150, 450, 1))
    # print(np_arr)
    arr = []
    count = 0
    for label, category in enumerate(categories):
        category_dir = os.path.join(image_dir, category) # к пути добавляется название папки
        for filename in os.listdir(category_dir): # для всех файлов в папке
            if filename.endswith('.jpg'):         # если это картинка(.jpg)
                pil_im = np.asarray(Image.open(os.path.join(category_dir, filename)))
                #print(pil_im.shape[0])
                if pil_im.shape[0] != 150 or pil_im.shape[1] != 150:
                    break
                reim = pil_im.reshape(pil_im.shape[0], pil_im.shape[1] * pil_im.shape[2], 1)
               # print("img: ", filename, '.Shape2: ', reim.shape)
               # print('d2.shape', d2.shape)
                #print("img: ", filename, '.Data2: ', reim)
                #print('d2_data', d2)
                np_arr = np.append(np_arr, reim, axis=2)
                
                #print('arr: ', np_arr.shape)
                #print('np_arr: ', np_arr)

    return np_arr



ds = set_dataset(df, train_image_dir)
print("np_arr: ", ds)
print("np_arr_shape: ", ds.shape)
print("np_arr_size: ", ds.size)
#set_dataset(df_t, test_image_dir)
#print(type(df))
#print(df)
#X, y = image_paths / 255.0, test_image_paths.astype(int)
#X_train, X_test, y_train, y_test = train_test_split(image_paths, image_paths, test_size=0.2, random_state=42)
# Загрузка данных MNIST
mnist = fetch_openml('mnist_784', version=1)
#print(mnist.data)
#X, y = mnist.data / 255.0, mnist.target.astype(int)

# Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Создание и обучение модели k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f'Accuracy of k-NN: {accuracy_knn:.4f}')