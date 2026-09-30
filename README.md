# End-to-End Applied Machine Learning & Deep Learning Pipelines

## 📌 Project Overview
This repository contains a comprehensive suite of machine learning and deep learning pipelines designed to tackle complex, real-world data challenges across multiple domains. The project rigorously evaluates advanced methodologies, emphasizing model interpretability, statistical significance, and adversarial robustness rather than relying solely on baseline accuracy[cite: 1, 2]. 

Key problem areas addressed include:
*   **Robust Regression:** Predicting housing prices while mitigating severe multicollinearity and outlier interference[cite: 2].
*   **Extreme Classification Imbalance:** Detecting credit card fraud within a dataset exhibiting a 577:1 imbalance ratio[cite: 2].
*   **Manifold Learning & Dimensionality Reduction:** Extracting 2D semantic topologies from high-dimensional image data[cite: 2].
*   **Unsupervised Cluster Ensembles:** Building stable consensus clusters on datasets without clear ground truth[cite: 2].
*   **Deep Learning Interpretability & Robustness:** Training convolutional networks with hyperparameter optimization, visualizing decision boundaries, and measuring degradation under adversarial attacks[cite: 2].

## 🛠️ Tech Stack & Tools
*   **Languages:** Python[cite: 1]
*   **Machine Learning:** Scikit-learn, XGBoost, LightGBM, imbalanced-learn (imblearn)[cite: 1, 2]
*   **Deep Learning:** PyTorch, TensorFlow[cite: 1]
*   **Dimensionality Reduction & Clustering:** UMAP, t-SNE, SciPy (Hierarchical Clustering)[cite: 2]
*   **Optimization & Tuning:** Optuna, Ray Tune, RandomizedSearchCV[cite: 1, 2]
*   **Interpretability & Robustness:** Grad-CAM, LIME, SHAP, Isotonic Regression, Fast Gradient Sign Method (FGSM)[cite: 1, 2]

## 🔬 Methodology

### 1. Feature Engineering & Robust Regression
*   Applied $log(1+x)$ transformations to highly skewed target variables and continuous predictors[cite: 2].
*   Generated degree 3 polynomial features and interaction terms, utilizing Tree-based feature importance to reduce the feature space to the most critical variables[cite: 2].
*   Replaced standard OLS with a `HuberRegressor` and regularized linear models (Elastic Net) to penalize extreme outliers linearly[cite: 2].

### 2. Cost-Sensitive Learning & Probability Calibration
*   Implemented an `imblearn.pipeline` to prevent data leakage during cross-validation when applying SMOTE and ADASYN[cite: 2].
*   Utilized cost-sensitive learning (`scale_pos_weight`) to penalize minority class misclassifications natively within XGBoost[cite: 2].
*   Calibrated output probabilities using Isotonic Regression to ensure predictions reflected true likelihoods[cite: 2].

### 3. Autoencoders & Semantic Extraction
*   Trained an undercomplete autoencoder with a 2-neuron bottleneck to compress MNIST data, achieving lower reconstruction MSE than standard Linear PCA[cite: 2].
*   Decoded the 2D latent space coordinates to map organic morphing trajectories of digits (e.g., stroke thickness and slant)[cite: 2].

### 4. Consensus Clustering
*   Evaluated K-Means, GMM, DBSCAN, and Agglomerative clustering using internal metrics (Silhouette, Davies-Bouldin) and external validation (Adjusted Rand Index)[cite: 2].
*   Constructed a Cluster Ensemble utilizing a Co-association Matrix to aggregate predictions from varied initializations into a single, stable consensus[cite: 2].

### 5. Adversarial Robustness & Explainable AI
*   Designed a Custom CNN with aggressive data augmentation and optimized learning rates/weight decay via Optuna[cite: 2].
*   Applied Grad-CAM to highlight spatial focal points, identifying "texture bias" and "background confusion" in misclassified samples[cite: 2].
*   Executed an FGSM adversarial attack ($\epsilon = 0.05$) to test decision boundary robustness, revealing a vulnerability gap between standard and adversarially trained models[cite: 2].
