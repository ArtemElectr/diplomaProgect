from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


#-----------------------------------------------------------------------------
import pandas as pd
import numpy as np

df = pd.read_csv('The Ultimate Cars Dataset 2024.csv',encoding='latin1')
#print(df)
#print(df.shape) # Возвращает кортеж, представляющий размерность фрейма данных.
#print(df.dtypes) # Он возвращает серию с типом данных каждого столбца.

# отбор числовых колонок
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
#print(numeric_cols)

# отбор нечисловых колонок
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
#print(non_numeric_cols)
#-----------------------------------------------------------------------------
iris = load_iris(as_frame=True)
X = iris.data[["sepal length (cm)", "sepal width (cm)"]]
y = iris.target


print(iris)
#print(iris)  # первые 30 колонок
#print(type(X))
print(X)
#print(type(y))
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0, test_size=0.2,) # а random_state – это параметр, который гарантирует одинаковое разделение при каждом запуске.
#print(X_train)
#print('!!!')
#print(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

clf = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))]
)

import matplotlib.pyplot as plt

from sklearn.inspection import DecisionBoundaryDisplay

_, axs = plt.subplots(ncols=2, figsize=(12, 5))
print('!!!')
print(iris.target_names)

for ax, weights in zip(axs, ("uniform", "distance")):
    clf.set_params(knn__weights=weights).fit(X_train, y_train)


    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_test,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
        shading="auto",
        alpha=0.5,
        ax=ax,
    )
    scatter = disp.ax_.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors="k")
    disp.ax_.legend(
        scatter.legend_elements()[0],
        iris.target_names,
        loc="lower left",
        title="Classes",
    )
    _ = disp.ax_.set_title(
        f"3-Class classification\n(k={clf[-1].n_neighbors}, weights={weights!r})"
    )

plt.show()