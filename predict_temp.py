import argparse
import datetime

import pandas as pd

from sklearn.preprocessing import StandardScaler
from lightgbm import Booster


SHIFTS = 5
TRAIN_COLUMNS_INPUTS = ['reg_pyl_zwrot_liw4', 'zawartosc_c', 'zawartosc_s', 'zawartosc_fe', 'zawartosc_fep', 'zawartosc_sp']
TRAIN_COLUMNS_HEAT = ['sumaryczna_moc_cieplna_odebrana', 'cieplo_pobrane_suma']
TRAIN_COLUMNS_AIR = ['went_zad_obr_1', 'went_zad_obr_2', 'went_zad_obr_3']
TRAIN_COLUMNS = TRAIN_COLUMNS_INPUTS + TRAIN_COLUMNS_HEAT + TRAIN_COLUMNS_AIR


def load_model(model_path : str):
    return Booster(model_file=model_path)


def remap_columns(dataset):
    return dataset.rename(columns={
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


def calculate_water_heat(dataset):
    dataset['temp_wody_zasil_srednia'] = dataset[['temp_wody_zasil_1', 'temp_wody_zasil_2']].mean(axis=1)

    dataset['cieplo_pobrane_7'] = (dataset['woda_powrotna_kz7'] - dataset['temp_wody_zasil_srednia']) * dataset[
        'woda_chlodzaca_kz7']
    dataset['cieplo_pobrane_8'] = (dataset['woda_powrotna_kz8'] - dataset['temp_wody_zasil_srednia']) * dataset[
        'woda_chlodzaca_kz8']
    dataset['cieplo_pobrane_9'] = (dataset['woda_powrotna_kz9'] - dataset['temp_wody_zasil_srednia']) * dataset[
        'woda_chlodzaca_kz9']
    dataset['cieplo_pobrane_10'] = (dataset['woda_powrotna_kz10'] - dataset['temp_wody_zasil_srednia']) * dataset[
        'woda_chlodzaca_kz10']
    dataset['cieplo_pobrane_11'] = (dataset['woda_powrotna_kz11'] - dataset['temp_wody_zasil_srednia']) * dataset[
        'woda_chlodzaca_kz11']
    dataset['cieplo_pobrane_12'] = (dataset['woda_powrotna_kz12'] - dataset['temp_wody_zasil_srednia']) * dataset[
        'woda_chlodzaca_kz12']
    dataset['cieplo_pobrane_13'] = (dataset['woda_powrotna_kz12'] - dataset['temp_wody_zasil_srednia']) * dataset[
        'woda_chlodzaca_kz13']
    dataset['cieplo_pobrane_15'] = (dataset['woda_powrotna_kz15'] - dataset['temp_wody_zasil_srednia']) * dataset[
        'woda_chlodzaca_kz15']

    dataset['cieplo_pobrane_suma'] = dataset['cieplo_pobrane_7'] + dataset['cieplo_pobrane_8'] + dataset[
        'cieplo_pobrane_9'] + dataset['cieplo_pobrane_10'] + dataset['cieplo_pobrane_11'] + dataset[
                                         'cieplo_pobrane_12'] + dataset['cieplo_pobrane_13'] + dataset[
                                         'cieplo_pobrane_15']
    return dataset


def calculate_fuels(dataset):
    dataset['reg_nadawy_koncentratu_suma'] = dataset['reg_nadawy_koncentratu_liw1'] + dataset[
        'reg_nadawy_koncentratu_liw2']
    dataset['zawartosc_c'] = dataset['reg_nadawy_koncentratu_suma'] * dataset['prob_corg']
    dataset['zawartosc_s'] = dataset['reg_nadawy_koncentratu_suma'] * dataset['prob_s']
    dataset['zawartosc_fe'] = dataset['reg_nadawy_koncentratu_suma'] * dataset['prob_fe']
    dataset['zawartosc_fep'] = dataset['reg_koncentrat_prazony_liw3'] * dataset['prazonka_fe']
    dataset['zawartosc_sp'] = dataset['reg_koncentrat_prazony_liw3'] * dataset['prazonka_s']
    return dataset


def load_dataset(dataset_path : str):
    dataset = pd.read_csv(dataset_path)

    # Cut out the TZ data because the data about temperature doesn't have it anyway
    dataset['czas'] = dataset['czas'].apply(lambda x: x[:-6])
    dataset['czas'] = pd.to_datetime(dataset['czas'], format='%Y-%m-%d %H:%M:%S')
    dataset = dataset.set_index('czas')

    dataset = remap_columns(dataset)
    dataset = calculate_water_heat(dataset)
    dataset = calculate_fuels(dataset)
    dataset = dataset[~dataset.index.duplicated(keep='first')]

    return dataset


def predict(args):
    model = load_model(args.model)
    dataset = load_dataset(args.parameters)
    filtered_dataset = dataset[dataset.index > args.last_measurement_time]

    x_test = filtered_dataset[TRAIN_COLUMNS]
    COLUMNS_TO_SCALE = ['reg_pyl_zwrot_liw4', 'zawartosc_c', 'zawartosc_s', 'zawartosc_fe',
                        'sumaryczna_moc_cieplna_odebrana', 'cieplo_pobrane_suma', 'went_zad_obr_1', 'went_zad_obr_2',
                        'went_zad_obr_3', 'zawartosc_fep', 'zawartosc_sp']

    for col in COLUMNS_TO_SCALE:
        scaler = StandardScaler()
        x_test.loc[:, col] = scaler.fit_transform(x_test[col].values.reshape(-1, 1))

    COLUMNS_TO_SHIFT = ['reg_pyl_zwrot_liw4', 'zawartosc_c', 'zawartosc_s', 'zawartosc_fe',
                        'sumaryczna_moc_cieplna_odebrana', 'cieplo_pobrane_suma', 'went_zad_obr_1', 'went_zad_obr_2',
                        'went_zad_obr_3', 'zawartosc_fep', 'zawartosc_sp']

    for col in COLUMNS_TO_SHIFT:
        for i in range(1, SHIFTS + 1):
            x_test.loc[:, '{}_shift_{}'.format(col, i)] = x_test[col].shift(i, fill_value=x_test[col][0])

    predictions = model.predict(x_test)
    print ('Estimated current temperature: ', args.last_measurement_value + sum(predictions))


def main():
    parser = argparse.ArgumentParser(description='Przewidywanie temperatury w piecu.')
    parser.add_argument('--parameters', help='Path to the file with parameters for prediction', required=True)
    parser.add_argument('--model', help='Path to existing model to make predictions', required=True)
    parser.add_argument('--last-measurement-time', type=datetime.datetime.fromisoformat, help='Time of last heat meassurement in ISO format', required=True)
    parser.add_argument('--last-measurement-value', type=float, help='Value of last heat meassurement', required=True)
    args = parser.parse_args()
    predict(args)


if __name__ == '__main__':
    main()