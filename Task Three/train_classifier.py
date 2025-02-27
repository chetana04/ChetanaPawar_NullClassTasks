import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('data.pickle', 'rb'))

expected_length = 42

data_filtered = []
labels_filtered = []
for sample, label in zip(data_dict['data'], data_dict['labels']):
    if len(sample) == expected_length:
        data_filtered.append(sample)
        labels_filtered.append(label)

if not data_filtered:
    raise ValueError("No samples with the expected length were found.")

data = np.array(data_filtered)
labels = np.array(labels_filtered)

print(f"Using {len(data)} samples out of {len(data_dict['data'])}")

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"{score * 100:.2f}% of samples were classified correctly!")

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
