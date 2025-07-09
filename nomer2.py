import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data
X = np.array([[30], [50], [70], [80], [100]])
y = np.array([150, 250, 300, 330, 400])

# Model
model = LinearRegression()
model.fit(X, y)

# Prediksi
predicted = model.predict(X)
print("Koefisien:", model.coef_)
print("Intercept:", model.intercept_)
print("Prediksi Harga:", predicted)

# Visualisasi
plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, predicted, color='red', label='Regresi Linear')
plt.xlabel('Luas (m²)')
plt.ylabel('Harga (juta)')
plt.title('Prediksi Harga Rumah')
plt.legend()
plt.grid()
plt.show()
