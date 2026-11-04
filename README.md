# machine-Learning-1
# Polynomial Regression with 5 Key Optimizers (Isolated Notebooks)

This repository contains an academic implementation of **Polynomial Regression** applied to a real-world climate dataset (**Earth Surface Temperature Data by Berkeley Earth**). 

The core objective of this project is to analyze, implement, and compare the performance of **5 different optimization algorithms (Optimizers)** in updating weights and minimizing the Mean Squared Error (MSE).

## 📊 Dataset & The Logic Behind the Full U-Shape Curve
Instead of using dummy or generated data, this project utilizes **real-world temperature records for Chicago**. 
* **The Cycle:** We extracted exactly **one full climatic cycle (12 months)** starting from July (Summer) through Winter and back to July.
* **The Feature Transformation:** Applied `PolynomialFeatures(degree=2)` after centering the timeline around zero.
* **The Result:** Because temperatures naturally drop in winter and rise in summer, the real dataset forces the model to capture a **perfect, continuous U-shaped quadratic curve** (Hot ➡️ Cold ➡️ Hot).

---

## 📋 Repository Structure
To ensure clean evaluation, the project is strictly divided into **5 separate Jupyter Notebooks (`.ipynb`)**, each dedicated to a single optimizer:

1. 📄 **`1_SGD_Optimizer.ipynb`**
   * Implements standard Stochastic Gradient Descent with a constant learning rate.
2. 📄 **`2_Mini_Batch_GD_Optimizer.ipynb`**
   * Implements Mini-Batch Gradient Descent via sequential batch updating (`partial_fit`).
3. 📄 **`3_Momentum_Optimizer.ipynb`**
   * Accelerates convergence by adding a velocity vector (`momentum=0.9`) to stabilize gradient steps.
4. 📄 **`4_AdaGrad_Optimizer.ipynb`**
   * Dynamically dampens or accelerates learning rates based on historically accumulated gradients (`learning_rate='adaptive'`).
5. 📄 **`5_Adam_Optimizer.ipynb`**
   * Leverages Adaptive Moment Estimation, merging both Momentum and adaptive scaling benefits.

---

## 🛠️ Optimization & Engineering Highlights
* **Gradient Explosion Prevention:** Scaled features using `StandardScaler` to bound polynomial outputs and secure smooth convergence.
* **Architecture Mapping:** Implemented complex optimizers (Momentum/Adam) via an identity-activated single-layer framework to maintain authentic regression boundaries without neural network non-linearities.

---

## 💻 Tech Stack
* **Language:** Python 3
* **Environment:** Kaggle Notebooks
* **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib
