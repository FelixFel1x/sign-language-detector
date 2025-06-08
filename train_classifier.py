import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

print(f"Anzahl der Samples in data_dict['data']: {len(data_dict['data'])}")
all_lengths_same = True
first_length = -1
if len(data_dict['data']) > 0:
    first_length = len(data_dict['data'][0])

for i, sample_features in enumerate(data_dict['data']):
    current_length = len(sample_features)
    if i < 10 or current_length != first_length: 
        print(f"Sample {i}: Länge = {current_length}")
    if current_length != first_length:
        all_lengths_same = False
        


if not all_lengths_same:
    print("FEHLER: Nicht alle Feature-Vektoren haben die gleiche Länge!")
else:
    print(f"Alle {len(data_dict['data'])} Feature-Vektoren haben die korrekte Länge von {first_length}.")

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
