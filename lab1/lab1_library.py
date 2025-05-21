import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import tree

# Функция для one hot encoding
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return res

# Загружаем данные
df = pd.read_csv("/Users/admin/Downloads/student/student-por.csv", sep=";")

# Разделяем данные на количественные и категориальные
df_numerical = df.select_dtypes(include=[np.number])
df_categorical = df.select_dtypes(exclude=[np.number])

# one hot encoding для категориальных признаков
features_to_encode = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
                      "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
for feature in features_to_encode:
    df_categorical = encode_and_bind(df_categorical, feature)

# Объединяем обработанные данные
new_df = pd.concat([df_numerical, df_categorical], axis=1)

# Строим тепловую карту корреляции
corr_matrix = new_df.corr().abs()
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f")
plt.title("Матрица корреляций (до удаления)")
# plt.show()

# Удаляем сильно коррелированные признаки (выше 0.95)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
new_df = new_df.drop(to_drop, axis=1)
print(f"Удалено {len(to_drop)} коррелирующих признаков: {to_drop}")

print(new_df)
# Строим тепловую карту после удаления
corr_matrix = new_df.corr().abs()
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f")
plt.title("Матрица корреляций (после удаления)")
# plt.show()

# Разделяем данные на признаки и целевую переменную
X = new_df.drop('G3', axis=1)
Y = new_df['G3']

# Обучаем дерево
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf = clf.fit(X, Y)

feature_importances = clf.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Сортируем и выводим топ 10 признаков
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Топ 10 признаков:")
print(importance_df.head(10))


