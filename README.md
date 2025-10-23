# 🎧 Moosic — Automating Playlist Creation with Unsupervised Learning

**Moosic** is a case study on how **unsupervised learning** can help automate playlist creation using **Spotify audio features**.  
We explore **K-Means**, **DBSCAN**, and **Agglomerative (Hierarchical)** clustering — with **PCA** for visual intuition — and compare results to build playlist “moods.”

This project was performed with *Olaf Bulas*.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Library-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Spotify API](https://img.shields.io/badge/Spotify-API-1DB954?logo=spotify&logoColor=white)](https://developer.spotify.com/documentation/web-api/)
[![Unsupervised Learning](https://img.shields.io/badge/Unsupervised%20Learning-Clustering-blueviolet)](https://scikit-learn.org/stable/modules/clustering.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

![Banner](banner.png)

---

## 🧠 Project Story

Moosic’s human curators can’t keep up with demand. We asked:

> *Can data science group songs into playlists automatically in a way people recognize as “similar mood or style”?*

This repo walks from **raw Spotify features** → **scaled, clustered groups** → **visual + quantitative evaluation**, mirroring the accompanying slides.

---

## 📁 Repository Structure

```plaintext
Moosic/
├── slides/
│   └── Moosic_Presentation.pptx
├── data/
│   ├── spotify_5000_songs.csv
│   └── spotify_5000_songs.cleaned.csv
├── notebooks/
│   ├── Spotify_5000_songs.ipynb
│   ├── 1_introduction_to_kmeans_Spotify_5000.ipynb
│   ├── 2_scaling_data_Spotify_5000.ipynb
│   ├── 3_analysing_k_means__choosing_k_Spotify_5000.ipynb
│   ├── 4_PCA_Spotify_5000.ipynb
│   ├── 5_DBSCAN_Spotify_5000.ipynb
│   └── 6_AgglomerativeClustering_Spotify_5000D.ipynb
├── reports/
│   ├── scaler_comparison_kmeans20.csv
│   ├── kmeans_sweep_results.csv
│   ├── dbscan_grid_results.csv
│   ├── agglomerative_grid_results.csv
│   ├── algo_comparison_results.csv
│   └── plots/
│       ├── histogram_energy.png
│       ├── correlation_heatmap.png
│       ├── scaler_comparison.png
│       ├── elbow_quantile.png
│       ├── silhouette_quantile.png
│       ├── pca2_kmeans20.png
│       ├── dbscan_best_pca.png
│       ├── dbscan_label_sizes.png
│       ├── dbscan_silhouette_heatmap.png
│       ├── dbscan_ch_heatmap.png
│       ├── dbscan_db_heatmap.png
│       ├── dbscan_noise_heatmap.png
│       ├── agglomerative_ward_pca.png
│       ├── agglomerative_cluster_sizes.png
│       ├── agglomerative_dendrogram_focused.png
│       ├── agg_silhouette_ward.png
│       ├── agg_ch_ward.png
│       └── agg_db_ward.png
```

---

## ▶️ Run Order (recommended)

1. **PCA** → `4_PCA_Spotify_5000.ipynb`  
   Builds intuition on variance & feature loadings; exports scree and cumulative plots.

2. **DBSCAN** → `5_DBSCAN_Spotify_5000.ipynb`  
   Runs eps × min_samples grid; produces silhouette & noise heatmaps, best PCA projection.

3. **Agglomerative** → `6_AgglomerativeClustering_Spotify_5000D.ipynb`  
   Compares linkages & cluster counts, builds dendrogram and PCA cluster map.

4. **Capstone** → `Spotify_5000_songs.ipynb`  
   Loads/cleans data, performs EDA, scaler comparison, k sweep, and final algorithm comparison.

> You can run the capstone alone; it will reuse previous results if CSVs exist.

---

## 🎵 Dataset

**5,000 Spotify tracks** with audio features:

- `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `liveness`, `speechiness`, `tempo`, `loudness`, `duration_ms`, `key`, `mode`, `time_signature`

**Example visualizations:**  
![Energy histogram](./reports/plots/histogram_energy.png)  
![Correlation heatmap](./reports/plots/correlation_heatmap.png)

---

## 🧪 Techniques

- **Scaling comparison** → `scaler_comparison_kmeans20.csv`  
  ![Scaler comparison](./reports/plots/scaler_comparison.png)

- **K-Means Elbow + Silhouette** → `kmeans_sweep_results.csv`  
  ![Elbow](./reports/plots/elbow_quantile.png) ![Silhouette](./reports/plots/silhouette_quantile.png)

- **DBSCAN** (density) → `dbscan_grid_results.csv`  
  ![DBSCAN best PCA](./reports/plots/dbscan_best_pca.png)

- **Agglomerative** (hierarchical) → `agglomerative_grid_results.csv`  
  ![Agg PCA](./reports/plots/agglomerative_ward_pca.png)

- **PCA Variance** → scree, cumulative, 2D/3D loadings plots

---

## 🧮 Metrics

| Metric | Optimize | Meaning |
|---|---|---|
| Silhouette | ↑ | Separation & cohesion |
| Calinski–Harabasz | ↑ | Compactness & separation |
| Davies–Bouldin | ↓ | Cluster distinctness |
| Inertia (K-Means) | ↓ | Within-cluster variance |

---

## 🧩 Findings

- **Features are skewed & correlated**, so scaling matters.  
- **QuantileScaler** produced more stable clusters.  
- **K-Means (k≈20)** → interpretable *mood/style* groups.  
- **DBSCAN** → identifies **outliers/niche blends**.  
- **Agglomerative** → provides an **explainable hierarchy**.

> Consolidated results: `algo_comparison_results.csv`

---

## 🖥️ Slides

**Presentation:** [`slides/Moosic_Presentation.pptx`](./slides/Moosic_Presentation.pptx)

---

## 🧰 Requirements

```bash
pip install pandas numpy scikit-learn matplotlib scipy
```

---

## 💬 Credits

- Data: Spotify Web API  
- ML: Scikit-learn (KMeans, DBSCAN, AgglomerativeClustering, PCA)  
- Visualization: Matplotlib  
- Educational adaptation by Moosic Data Team

---

## 💡 Next Steps

- Add **genre** or **lyrics sentiment** for hybrid mood detection  
- Build a **Streamlit** app to browse clusters with song previews  
- Integrate with **Spotify API** for playlist generation

---

## 🎵 Final Thought

Even without labels, **unsupervised learning** reveals how songs *feel* similar.  
It won’t replace human taste — but it can **inspire** playlists you’d never think to make.
