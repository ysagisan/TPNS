import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Функция для one hot encoding чтобы отформотировать категориальные данные в число
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
plt.show()

# Удаляем сильно коррелированные признаки (выше 0.95)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
new_df = new_df.drop(to_drop, axis=1)
print(f"Удалено {len(to_drop)} коррелирующих признаков: {to_drop}")

# Строим тепловую карту после удаления
corr_matrix = new_df.corr().abs()
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f")
plt.title("Матрица корреляций (после удаления)")
plt.show()

# Выбираем целевую переменную - итоговую оценку G3
y = df['G3']

# Функция для расчета энтропии
def calculate_entropy(y):
    probs = np.bincount(y) / len(y)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# Функция для расчета Information Gain
def information_gain(X, y, feature):
    total_entropy = calculate_entropy(y)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_entropy = sum((counts[i] / len(y)) * calculate_entropy(y[X[feature] == v]) for i, v in enumerate(values))
    return total_entropy - weighted_entropy

# Функция для расчета Split Information
def split_information(X, feature):
    values, counts = np.unique(X[feature], return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

# Вычисляем Gain Ratio для всех признаков
def gain_ratio(X, y):
    gain_ratios = {}
    for feature in X.columns:
        ig = information_gain(X, y, feature)
        si = split_information(X, feature)
        gr = ig / si if si != 0 else 0
        gain_ratios[feature] = gr
    return pd.Series(gain_ratios).sort_values(ascending=False)


# Вычисляем Gain Ratio
gain_ratio_scores = gain_ratio(new_df, y)

# Выбираем топ 10 признаков
top_features = gain_ratio_scores.head(10).index.tolist()
new_df_final = new_df[top_features]

print("Топ 10 признаков по Gain Ratio:")
print(gain_ratio_scores.head(10))
