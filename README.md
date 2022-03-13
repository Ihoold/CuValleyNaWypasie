# CuValley - NaWypasie 
- Projekt: sztuczny analizator temperatury w piecu
- Hackaton: cuvalley.com

# Obsługa
pip install lightgbm, pandas, scikit-learn

Przykładowe użycie predyktora :
```
python predict_temp.py --parameters %cd%\zadanie3\dane_test\avg_from_2022_01_31_00_00_00_to_2022_01_31_23_59_00 --model %cd%\zadanie3\lgb_regressor.txt --last-measurement-time "2022-01-31 23:00:00" --last-measurement-value 1309
```