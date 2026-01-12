import dill
import pandas as pd
import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler


# Шаг 1. Заполнение пропущенных значений в фиче device_brand на основе фичи device_os
def completion_device_brand(ga_final):
    ga_final.loc[ga_final.device_brand.isna(), 'device_brand'] = ga_final.loc[ga_final.device_os.isna(), 'device_os'].apply(lambda x: 'Apple' if x == 'iOS'
                                                                                                                  else ('Microsoft' if x == 'Windows'
                                                                                                                  else 'Other'))
    return ga_final


# Шаг 2. Фичи device_model, device_os и utm_keyword считаем неинформативными и удаляем.
def filter_data1(ga_final):
    columns_to_drop = [
        'device_model',
        'device_os',
        'utm_keyword',
    ]
    return ga_final.drop(columns_to_drop, axis=1)


# Шаг 3. На основе фичей visit_date и visit_time создаем новые фичи: месяц, день недели, утро, день, вечер, ночь типа int
def time_features(ga_final):
    ga_final.loc[:, 'date'] = pd.to_datetime(ga_final['visit_date'] + ' ' + ga_final['visit_time'], utc=True)
    ga_final.loc[:, 'month'] = ga_final['date'].dt.month
    ga_final.loc[:, 'day_of_week'] = ga_final['date'].dt.dayofweek

    ga_final.loc[:, 'time_of_day'] = ga_final['date'].dt.time
    ga_final.loc[:, 'is_morning'] = (ga_final['time_of_day'] >= pd.Timestamp('06:00:00').time()) & (
                ga_final['time_of_day'] < pd.Timestamp('12:00:00').time())
    ga_final.loc[:, 'is_afternoon'] = (ga_final['time_of_day'] >= pd.Timestamp('12:00:00').time()) & (
                ga_final['time_of_day'] < pd.Timestamp('18:00:00').time())
    ga_final.loc[:, 'is_evening'] = (ga_final['time_of_day'] >= pd.Timestamp('18:00:00').time()) & (
                ga_final['time_of_day'] < pd.Timestamp('23:59:59').time())
    ga_final.loc[:, 'is_night'] = (ga_final['time_of_day'] >= pd.Timestamp('00:00:00').time()) & (
                ga_final['time_of_day'] < pd.Timestamp('06:00:00').time())
    ga_final['is_morning'] = ga_final['is_morning'].astype(int)
    ga_final['is_afternoon'] = ga_final['is_afternoon'].astype(int)
    ga_final['is_evening'] = ga_final['is_evening'].astype(int)
    ga_final['is_night'] = ga_final['is_night'].astype(int)
    return ga_final


# Шаг 4. Объединим подобные значения фичи device_browser и создадим дополнительную фичу device_browser_new
def device_browser_new_features(ga_final):
    ga_final.loc[:, 'device_browser_new'] = ga_final.device_browser.apply(lambda x: x.lower().split(' ')[0])

    # Заменим значения количество которых меньше 10000 на 'Other'
    counts = ga_final['device_browser_new'].value_counts()
    ga_final.loc[:, 'device_browser_new'] = ga_final['device_browser_new'].apply(
        lambda x: x if counts[x] > 10000 else 'Other')
    return ga_final


# Шаг 5. Создадим новые фичи geo_city_new, utm_campaign_new, utm_adcontent_new, utm_source_new
       # В фиче geo_city заменим значения, количество которых меньше 1000 на 'Other'
       # В фиче utm_campaign заменим значения, количество которых меньше 500 на 'Other'
       # В фиче utm_adcontent заменим значения, количество которых меньше 100 на 'Other'
       # В фиче utm_source заменим значения, количество которых меньше 100 на 'Other'

def create_features(ga_final):
    counts1 = ga_final['geo_city'].value_counts()
    counts2 = ga_final['utm_campaign'].value_counts()
    counts3 = ga_final['utm_adcontent'].value_counts()
    counts4 = ga_final['utm_source'].value_counts()
    ga_final.loc[:, 'geo_city_new'] = ga_final['geo_city'].apply(
        lambda x: x if counts1[x] > 1000 else 'Other')
    ga_final.loc[:, 'utm_campaign_new'] = ga_final['utm_campaign'].apply(
        lambda x: x if x in counts2 and counts2[x] > 500 else 'Other')
    ga_final.loc[:, 'utm_adcontent_new'] = ga_final['utm_adcontent'].apply(
        lambda x: x if x in counts3 and counts3[x] > 100 else 'Other')
    ga_final.loc[:, 'utm_source_new'] = ga_final['utm_source'].apply(
        lambda x: x if x in counts4 and counts4[x] > 100 else 'Other')
    return ga_final


# Шаг 6. Фильтрация датафрейма
def filter_data2(ga_final):
    columns_to_drop = [
        'session_id',
        'client_id',
        'visit_date',
        'visit_time',
        'visit_number',
        'utm_source',
        'utm_campaign',
        'utm_adcontent',
        'device_browser',
        'date',
        'time_of_day',
        'device_screen_resolution',
        'device_browser',
        'geo_city',
    ]

    return ga_final.drop(columns_to_drop, axis=1)


def main():

    print('Final Pipeline')

    ga_final = pd.read_csv('data/ga_final.csv')

    # Шаг 7. Балансировка классов целевой переменной
    X = ga_final.drop('event_value', axis=1)
    y = ga_final['event_value']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df['event_value'] = y_resampled

    numerical_features = make_column_selector(dtype_include=['int32', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    # Шаг 8. Масштабирование в StandardScaler
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Шаг 9. Заполнение пропусков в категориальных переменных и кодирование OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int))
    ])

    transformer = Pipeline(steps=[
        ('device_brand', FunctionTransformer(completion_device_brand)),
        ('filter1', FunctionTransformer(filter_data1)),
        ('time_features', FunctionTransformer(time_features)),
        ('device_browser', FunctionTransformer(device_browser_new_features)),
        ('create_features', FunctionTransformer(create_features)),
        ('filter2', FunctionTransformer(filter_data2)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = (
        LogisticRegression(C=10, class_weight='balanced', max_iter=1000),
        RandomForestClassifier(class_weight='balanced')
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('transform', transformer),
            ('preproc', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    with open('rent_a_car_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Rent a car prediction model',
                'author': 'Kirill Zhelezko',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file,
            recurse=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()