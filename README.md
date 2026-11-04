# machine-Learning-1
# Polynomial Regression with 5 Key Optimizers

This repository contains a beginner-friendly implementation of **Polynomial Regression** applied to a real-world dataset (**California Housing Prices**). The core objective of this project is to analyze, implement, and compare the performance of **5 different optimization algorithms (Optimizers)** in updating weights and minimizing the Mean Squared Error (MSE).

## 📋 Project Requirements & Specifications

As requested, the project has been structured into separate, isolated code blocks (cells) to independently evaluate each optimizer on a non-linear (Polynomial) dataset.

### 1. Dataset & Preprocessing
* **Dataset:** California Housing Prices (Real Dataset).
* **Feature Transformation:** Applied `PolynomialFeatures(degree=2)` to capture the non-linear relationship between *Median Income* and *Median House Value*.
* **Feature Scaling:** Utilized `StandardScaler` to normalize the data, ensuring stable and faster convergence for all gradient-based optimizers.

### 2. Implemented Optimizers (5 Models)
The project evaluates the following 5 optimization techniques:
1. **GD / SGD (Stochastic Gradient Descent):** Updates weights based on single random samples with a constant learning rate.
2. **Mini-Batch GD:** Updates weights using smaller subsets of data (implemented sequentially via mini-batches).
3. **GD with Momentum:** Accelerates gradient descent by accumulating a velocity vector (`momentum=0.9`) to prevent getting stuck in local minima.
4. **AdaGrad / Adaptive Learning Rate:** Dynamically adapts the learning rate parameter-by-parameter based on historical gradients.
5. **Adam (Adaptive Moment Estimation):** Combines the advantages of both Momentum and RMSProp/AdaGrad for optimal performance.

---

## 📊 Evaluation & Results Summary

After training each model independently, the **Mean Squared Error (MSE)** was calculated for comparison. 

* *Note on Visualization:* The final plot beautifully illustrates a **clear quadratic polynomial curve** (representing the accelerating right arm of a parabolic $U$-shape), perfectly capturing how housing prices scale non-linearly with income.

---

## 🛠️ Technology Stack
* **Language:** Python 3
* **Environment:** Kaggle Notebook
* **Libraries:** Pandas, NumPy, Matplotlib, Scikit-Learn
