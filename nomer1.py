import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data
data = np.array([
    [1, 1],
    [4, 1],
    [6, 1],
    [1, 2],
    [2, 3],
    [5, 3],
    [2, 5],
    [3, 5],
    [2, 6],
    [3, 8]
])

# Membuat dataframe
df = pd.DataFrame(data, columns=['x', 'y'])

# Menjalankan K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['x', 'y']])

# Centroid
centroids = kmeans.cluster_centers_

# Output hasil
print("Centroid akhir:")
print(centroids)
print("\nLabel cluster tiap data:")
print(df)

# Visualisasi
colors = ['red', 'green', 'blue']
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {i}', color=colors[i])

plt.scatter(centroids[:,0], centroids[:,1], s=200, c='black', marker='X', label='Centroid')
plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()
