BASE_PATH = 'zadanie-3'

import glob
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, Booster
from os import path

"""# Przygotowanie danych"""

file_list = glob.glob(path.join(BASE_PATH, 'dane', '*'))

datasets = []

for file_path in file_list:
    partial_dataset = pd.read_csv(file_path)
    datasets.append(partial_dataset)

dataset = pd.concat(datasets)
# Cut out the TZ data because the data about temperature doesn't have it anyway
dataset['czas'] = dataset['czas'].apply(lambda x: x[:-6])
dataset['czas'] = pd.to_datetime(dataset['czas'], format='%Y-%m-%d %H:%M:%S')
dataset = dataset.set_index('czas')

dataset = dataset.rename(columns={
    '001fcx00211.pv': 'reg_nadawy_koncentratu_liw1',
    '001fcx00221.pv': 'reg_nadawy_koncentratu_liw2',
    '001fcx00231.pv': 'reg_koncentrat_prazony_liw3', 
    '001fcx00241.pv': 'reg_pyl_zwrot_liw4',
    '001fir01307.daca.pv': 'woda_chlodzaca_kz7',
    '001fir01308.daca.pv': 'woda_chlodzaca_kz8',
    '001fir01309.daca.pv': 'woda_chlodzaca_kz9',
    '001fir01310.daca.pv': 'woda_chlodzaca_kz10',
    '001fir01311.daca.pv': 'woda_chlodzaca_kz11',
    '001fir01312.daca.pv': 'woda_chlodzaca_kz12',
    '001fir01313.daca.pv': 'woda_chlodzaca_kz13',
    '001fir01315.daca.pv': 'woda_chlodzaca_kz15',
    '001nir0szr0.daca.pv': 'sumaryczna_moc_cieplna_odebrana',
    '001tir01357.daca.pv': 'woda_powrotna_kz7',
    '001tir01358.daca.pv': 'woda_powrotna_kz8', 
    '001tir01359.daca.pv': 'woda_powrotna_kz9',
    '001tir01360.daca.pv': 'woda_powrotna_kz10', 
    '001tir01361.daca.pv': 'woda_powrotna_kz11', 
    '001tir01362.daca.pv': 'woda_powrotna_kz12',
    '001tir01363.daca.pv': 'woda_powrotna_kz13', 
    '001tir01365.daca.pv': 'woda_powrotna_kz15', 
    '001tix01063.daca.pv': 'temp_pod_wymurowka_1',
    '001tix01064.daca.pv': 'temp_pod_wymurowka_2',
    '001tix01065.daca.pv': 'temp_pod_wymurowka_3', 
    '001tix01066.daca.pv': 'temp_pod_wymurowka_4',
    '001tix01067.daca.pv': 'temp_pod_wymurowka_5', 
    '001tix01068.daca.pv': 'temp_pod_wymurowka_6', 
    '001tix01069.daca.pv': 'temp_pod_wymurowka_7',
    '001tix01070.daca.pv': 'temp_pod_wymurowka_8', 
    '001tix01071.daca.pv': 'temp_pod_wymurowka_9', 
    '001tix01072.daca.pv': 'temp_pod_wymurowka_10',
    '001tix01073.daca.pv': 'temp_pod_wymurowka_11', 
    '001tix01074.daca.pv': 'temp_pod_wymurowka_12', 
    '001tix01075.daca.pv': 'temp_pod_wymurowka_13',
    '001tix01076.daca.pv': 'temp_pod_wymurowka_14',
    '001tix01077.daca.pv': 'temp_pod_wymurowka_15', 
    '001tix01078.daca.pv': 'temp_pod_wymurowka_16',
    '001tix01079.daca.pv': 'temp_pod_wymurowka_17', 
    '001tix01080.daca.pv': 'temp_pod_wymurowka_18',
    '001tix01081.daca.pv': 'temp_pod_wymurowka_19',
    '001tix01082.daca.pv': 'temp_pod_wymurowka_20',
    '001tix01083.daca.pv': 'temp_pod_wymurowka_21', 
    '001tix01084.daca.pv': 'temp_pod_wymurowka_22',
    '001tix01085.daca.pv': 'temp_pod_wymurowka_23', 
    '001tix01086.daca.pv': 'temp_pod_wymurowka_24', 
    '001txi01153.daca.pv': 'temp_na_kol_kanal_1_34',
    '001txi01154.daca.pv': 'temp_na_kol_kanal_35_68', 
    '001uxm0rf01.daca.pv': 'went_zad_obr_1',
    '001uxm0rf02.daca.pv': 'went_zad_obr_2',
    '001uxm0rf03.daca.pv': 'went_zad_obr_3',
    '037tix00254.daca.pv': 'temp_wody_zasil_1', 
    '037tix00264.daca.pv': 'temp_wody_zasil_2' 
})

data_temp_zuz = pd.read_csv('/content/drive/MyDrive/CuValley/zadanie-3/temp_zuz.csv', sep=';')
data_temp_zuz['Czas'] = pd.to_datetime(data_temp_zuz['Czas'])
data_temp_zuz = data_temp_zuz.set_index('Czas')

"""# Ciepło odebrane przez wodę"""

dataset['temp_wody_zasil_srednia'] = dataset[['temp_wody_zasil_1', 'temp_wody_zasil_2']].mean(axis=1)

dataset['cieplo_pobrane_7'] = (dataset['woda_powrotna_kz7'] - dataset['temp_wody_zasil_srednia'])*dataset['woda_chlodzaca_kz7']
dataset['cieplo_pobrane_8'] = (dataset['woda_powrotna_kz8'] - dataset['temp_wody_zasil_srednia'])*dataset['woda_chlodzaca_kz8']
dataset['cieplo_pobrane_9'] = (dataset['woda_powrotna_kz9'] - dataset['temp_wody_zasil_srednia'])*dataset['woda_chlodzaca_kz9']
dataset['cieplo_pobrane_10'] = (dataset['woda_powrotna_kz10'] - dataset['temp_wody_zasil_srednia'])*dataset['woda_chlodzaca_kz10']
dataset['cieplo_pobrane_11'] = (dataset['woda_powrotna_kz11'] - dataset['temp_wody_zasil_srednia'])*dataset['woda_chlodzaca_kz11']
dataset['cieplo_pobrane_12'] = (dataset['woda_powrotna_kz12'] - dataset['temp_wody_zasil_srednia'])*dataset['woda_chlodzaca_kz12']
dataset['cieplo_pobrane_13'] = (dataset['woda_powrotna_kz12'] - dataset['temp_wody_zasil_srednia'])*dataset['woda_chlodzaca_kz13']
dataset['cieplo_pobrane_15'] = (dataset['woda_powrotna_kz15'] - dataset['temp_wody_zasil_srednia'])*dataset['woda_chlodzaca_kz15']

dataset['cieplo_pobrane_suma'] = dataset['cieplo_pobrane_7'] + dataset['cieplo_pobrane_8'] + dataset['cieplo_pobrane_9'] + dataset['cieplo_pobrane_10'] + dataset['cieplo_pobrane_11'] + dataset['cieplo_pobrane_12'] + dataset['cieplo_pobrane_13'] + dataset['cieplo_pobrane_15']

"""# Składniki dostarczane do pieca"""

dataset['reg_nadawy_koncentratu_suma'] = dataset['reg_nadawy_koncentratu_liw1'] + dataset['reg_nadawy_koncentratu_liw2']
dataset['zawartosc_c'] = dataset['reg_nadawy_koncentratu_suma'] * dataset['prob_corg']
dataset['zawartosc_s'] = dataset['reg_nadawy_koncentratu_suma'] * dataset['prob_s']
dataset['zawartosc_fe'] = dataset['reg_nadawy_koncentratu_suma'] * dataset['prob_fe']
dataset['zawartosc_fep'] = dataset['reg_koncentrat_prazony_liw3'] * dataset['prazonka_fe']
dataset['zawartosc_sp'] = dataset['reg_koncentrat_prazony_liw3'] * dataset['prazonka_s']

"""# Połączone dane

"""

# Drop duplicated indices (e.g. on the time zone shift)
dataset = dataset[~dataset.index.duplicated(keep='first')]
data_temp_zuz = data_temp_zuz[~data_temp_zuz.index.duplicated(keep='first')]

result = pd.merge(dataset, data_temp_zuz, how='left', left_index=True, right_index=True)

result['temp_zuz_interpolated'] = result['temp_zuz'].interpolate(method='slinear', order=1)
# We need to fill values before last temperature is omitted by the left join
result['temp_zuz_interpolated'] = result['temp_zuz_interpolated'].fillna(data_temp_zuz['temp_zuz'][-1])

result['delta_temp_zuz_interpolated'] = result['temp_zuz_interpolated'].diff().fillna(0)

"""# Podział na zbiory"""

import math

TEST_SIZE = 0.1
TRAIN_COLUMNS_INPUTS = ['reg_pyl_zwrot_liw4', 'zawartosc_c', 'zawartosc_s', 'zawartosc_fe', 'zawartosc_fep', 'zawartosc_sp']
TRAIN_COLUMNS_HEAT = ['sumaryczna_moc_cieplna_odebrana', 'cieplo_pobrane_suma']
TRAIN_COLUMNS_AIR = ['went_zad_obr_1', 'went_zad_obr_2', 'went_zad_obr_3']
TRAIN_COLUMNS = TRAIN_COLUMNS_INPUTS + TRAIN_COLUMNS_HEAT + TRAIN_COLUMNS_AIR

test_split = math.floor((1-TEST_SIZE) * len(result))

X_train = result[TRAIN_COLUMNS][:test_split]
Y_train = result['temp_zuz_interpolated'][:test_split]
Y_train_delta = result['delta_temp_zuz_interpolated'][:test_split]

X_test = result[TRAIN_COLUMNS][test_split:]
Y_test = result['temp_zuz_interpolated'][test_split:]

"""# Preprocessing"""

COLUMNS_TO_SCALE = ['reg_pyl_zwrot_liw4', 'zawartosc_c', 'zawartosc_s', 'zawartosc_fe', 'sumaryczna_moc_cieplna_odebrana', 'cieplo_pobrane_suma', 'went_zad_obr_1', 'went_zad_obr_2', 'went_zad_obr_3', 'zawartosc_fep', 'zawartosc_sp']

for col in COLUMNS_TO_SCALE:
    scaler = StandardScaler()
    X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))

COLUMNS_TO_SHIFT = ['reg_pyl_zwrot_liw4', 'zawartosc_c', 'zawartosc_s', 'zawartosc_fe', 'sumaryczna_moc_cieplna_odebrana', 'cieplo_pobrane_suma', 'went_zad_obr_1', 'went_zad_obr_2', 'went_zad_obr_3', 'zawartosc_fep', 'zawartosc_sp']
SHIFTS = 5

for col in COLUMNS_TO_SHIFT:
    for i in range(1, SHIFTS+1):
        X_train['{}_shift_{}'.format(col, i)] = X_train[col].shift(i, fill_value=X_train[col][0])
        X_test['{}_shift_{}'.format(col, i)] = X_test[col].shift(i, fill_value=X_test[col][0])

"""# Model bazowy (naiwny)"""

Y_baseline = result['temp_zuz_interpolated'][test_split-1:-1]
mean_squared_error(Y_test, Y_baseline)

"""# Model LGBM zmiany temperatury"""

def lgb_model(X_train, Y_train, X_test):
    regressor = LGBMRegressor(
        max_depth=6,
        num_leaves=30,
        n_estimators=100,
        learning_rate=0.05,
        min_child_samples=100,
        objective='rmse'
    )
    regressor = regressor.fit(X_train, Y_train)
    return regressor, regressor.predict(X_test)

model, Y_pred_delta = lgb_model(X_train, Y_train_delta, X_test)
print('LGBM regression test error: ', mean_squared_error(Y_test[1:], Y_test[:-1] + Y_pred_delta[:-1]))
model.booster_.save_model(path.join(BASE_PATH, 'lgb_regressor.txt'), num_iteration=model.booster_.best_iteration)