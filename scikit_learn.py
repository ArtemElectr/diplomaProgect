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

arr = np.array([[[247, 247, 245],
        [247, 247, 245],
        [248, 248, 246],
        [248, 248, 248],
        [247, 247, 245],
        [247, 247, 245]],
       [[77, 76, 74],
        [47, 46, 44],
        [43, 42, 40],
        [22, 17,  14],
        [55, 50, 47],
        [39, 34, 31]]])

#print(arr)
new_arr = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])
#print(new_arr)
def set_dataset(df1, image_dir):
    np_arr = np.empty((150, 450))
    arr = []
    count = 0
    for label, category in enumerate(categories):
        category_dir = os.path.join(image_dir, category) # к пути добавляется название папки
        for filename in os.listdir(category_dir): # для всех файлов в папке
            if filename.endswith('.jpg'):         # если это картинка(.jpg)
                #im = iio.imread(os.path.join(category_dir, filename))
                print(filename)
                #print(im)
               # print(im.shape)
                pil_im = np.asarray(Image.open(os.path.join(category_dir, filename)))

                reim = pil_im.reshape(pil_im.shape[0], pil_im.shape[1] * pil_im.shape[2])
                print("New: ", reim.shape)
                np_arr = np.append(np_arr, reim,)
                print('arr: ', np_arr.shape)

                count+=1
                print('count: ', count)
                if count == 2:
                    break
                #new_frame = pd.DataFrame(np.reshape(im,(im.size,)))

                ##df1.to_csv('intel_images.csv')
                break
    np_arr =np.append(np_arr,arr)
    print("np_arr: ", np_arr)
    print("np_arr_shape: ", np_arr.shape)
    #np.savetxt('arrIm.csv', np_arr)
    #print('np_arr: ', np_arr)
    #print('pd_read:', pd.read_csv('arrIm.csv', encoding='utf-8'))


set_dataset(df, train_image_dir)
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