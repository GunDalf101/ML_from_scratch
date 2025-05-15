# 🧠 Machine Learning Algorithms from Scratch

Welcome to **ML From Scratch** — a collection of the most common machine learning algorithms, implemented purely in Python without using machine learning libraries like scikit-learn or TensorFlow.

This project is designed to help developers and students understand how popular ML algorithms work at a fundamental level — by building them from the ground up.

## 📚 Algorithms Included

| Algorithm               | Type                           | Status | Link |
|-------------------------|--------------------------------|--------|------|
| Linear Regression       | Supervised (regression)        | ✅ Done | [Linear Regression](./linear_regression) |
| Logistic Regression     | Supervised (classification)    | ✅ Done | [Logistic Regression](./logistic_regression) |
| K-Nearest Neighbors     | Supervised (classification)    | ✅ Done | [KNN](./knn) |
| Naive Bayes             | Supervised (classification)    | ✅ Done  | [Naive Bayes](./naive_bayes) |
| Decision Tree           | Supervised (classification)    | ✅ Done | [Decision Tree](./decision_tree) |
| Random Forest           | Supervised (classification)    | ✅ Done | [Random Forest](./random_forest) |
| Support Vector Machine  | Supervised (classification)    | ✅ Done | [SVM](./svm) |
| Principal Component Analysis | Unsupervised (dim. reduction) | ✅ Done | [PCA](./pca) |
| K-Means Clustering      | Unsupervised (clustering)      | ✅ Done| [K-Means](./kmeans) |
| Single Layer Perceptron              | Supervised (NN binary)         | ✅ Done | [Perceptron](./perceptron) |
| Neural Network (MLP)         | Supervised (NN multi-class)        | ✅ Done  | [Neural Network](./neural_network) |
| L1 & L2 Regularization       | Optimization                       | 📝 Planned  | [Regularization](./regularization) |
| Dropout                      | Regularization                     | 📝 Planned  | [Dropout](./dropout) |
| Gradient Checking            | Debugging Tool                     | 📝 Planned  | [Gradient Checking](./gradient_checking) |
| Hierarchical Clustering      | Unsupervised (clustering)          | 📝 Planned  | [Hierarchical Clustering](./hierarchical_clustering) |
| DBSCAN                       | Unsupervised (density clustering)  | 📝 Planned  | [DBSCAN](./dbscan) |
| Autoencoder (basic)          | Unsupervised (NN)                  | 📝 Planned  | [Autoencoder](./autoencoder) |
| AdaBoost                     | Ensemble (boosting)                | ✅ Done  | [AdaBoost](./adaboost) |
| Gradient Boosting            | Ensemble (boosting)                | 📝 Planned  | [Gradient Boosting](./gradient_boosting) |
| Recurrent Neural Network (RNN) | Supervised (sequential)          | 📝 Planned  | [RNN](./rnn) |
| Long Short-Term Memory (LSTM) | Supervised (sequential)           | 📝 Planned  | [LSTM](./lstm) |
| Convolutional Neural Network (CNN) | Supervised (image)         | 📝 Planned  | [CNN](./cnn) |
| Transformer (Basic)          | Supervised (attention mechanism)   | 📝 Planned  | [Transformer](./transformer) |

## 🔧 Installation

```bash
git clone https://github.com/GunDalf101/ML_from_scratch.git
cd ML_from_scratch
pip install -r requirements.txt
```

Requirements include:
- numpy
- matplotlib (for demos)
- seaborn (for demos)
- pandas (for demos)
- jupyterlab (for demos)
- scikit-learn (only for comparing results — not used in implementations)

## 🧪 How to Use

Each algorithm has:
- 📄 Its own Python implementation file
- 📊 A demo Jupyter notebook with visualizations and explanations
- 📘 A mini README explaining theory + usage

Example:
```bash
cd decision_tree
jupyter notebook demo.ipynb
```

## 💡 Goals of the Project

- Understand ML theory through implementation
- Master math concepts like gradients, dot products, entropy, etc.
- Improve Python coding skills
- Create an impressive portfolio piece

## 🤖 Demo Previews

*Add some images/gifs of decision boundaries, PCA projections, etc.*

## ✍️ Author

Made with magic and curiosity by GunDalf