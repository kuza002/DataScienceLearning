import joblib
import pandas as pd
from math import isnan
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def filter_data(df):
    """Удаляет ненужные столбцы из датафрейма."""
    df = df.copy()
    columns_to_drop = [
        'id', 'url', 'region', 'region_url', 'price',
        'manufacturer', 'image_url', 'description',
        'posting_date', 'lat', 'long'
    ]
    return df.drop(columns=columns_to_drop, axis=1)


def create_short_model(df):
    """Создает новый столбец с первым словом из модели автомобиля."""
    df = df.copy()

    def get_first_word(x):
        if isinstance(x, str):
            return x.lower().split(' ')[0]
        return x  # Возвращаем x, если это не строка или NaN

    df['short_model'] = df['model'].apply(get_first_word)
    return df.drop(columns=['model'])


def create_age_categories(df):
    """Создает категории возраста автомобиля."""
    df = df.copy()

    def get_category(x):
        if isnan(x):
            return x
        if int(x) > 2013:
            return 'new'
        elif int(x) < 2006:
            return 'old'
        else:
            return 'average'

    df['age_category'] = df['year'].apply(get_category)
    return df


# Загрузка данных
df = pd.read_csv('homework.csv')

# Определение пайплайнов для трансформации признаков
features_transformer = Pipeline(steps=[
    ('filter', FunctionTransformer(filter_data)),
    ('short_model', FunctionTransformer(create_short_model)),
    ('age_category', FunctionTransformer(create_age_categories))
])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Объединение трансформеров в ColumnTransformer
columns_transformer = ColumnTransformer(transformers=[
    ('numeric', numeric_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
    ('categorical', categorical_transformer, make_column_selector(dtype_include=['object']))
])

# Разделение данных на признаки и целевую переменную
X = df.drop('price_category', axis=1)
y = df['price_category']

# Определение моделей для обучения
models = [
    LogisticRegression(solver='liblinear'),
    RandomForestClassifier(),
    SVC()
]

# Обучение моделей и выбор лучшей
best_score = 0.0
best_pipe = None

for model in models:
    pipe = Pipeline(steps=[
        ('features_transformer', features_transformer),
        ('columns_transformer', columns_transformer),
        ('classifier', model)
    ])

    score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')

    print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
    if score.mean() > best_score:
        best_score = score.mean()
        best_pipe = pipe

# Вывод информации о лучшей модели
print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

# Сохранение лучшей модели
joblib.dump(best_pipe, 'loan_pipe.pkl')
