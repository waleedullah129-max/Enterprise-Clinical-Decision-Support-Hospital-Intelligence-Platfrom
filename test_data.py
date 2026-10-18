import pandas as pd

mimic = pd.read_csv('data/mimic/hosp/patients.csv.gz')
print('MIMIC patients:', mimic.shape)

synthea = pd.read_csv('data/synthea/patients.csv')
print('Synthea patients:', synthea.shape)

asclepius = pd.read_csv('data/asclepius/synthetic.csv')
print('Asclepius notes:', asclepius.shape)