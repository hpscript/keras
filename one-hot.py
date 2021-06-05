from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

cities = ['London', 'Berlin', 'Berlin', 'New York', 'London']

encoder = LabelEncoder()
city_labels = encoder.fit_transform(cities)
encoder = OneHotEncoder(sparse=False)
city_labels = city_labels.reshape((5, 1))
array = encoder.fit_transform(city_labels)
print(array)