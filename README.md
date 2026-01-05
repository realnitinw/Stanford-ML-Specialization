#  Stanford Machine Learning Specialization Vault
This repository contains my technical implementations and lab assignments from the **Stanford Machine Learning Specialization** taught by Andrew Ng.

###  Technical Scope
As a Computer Science student at MSIT (Class of 2029), I have used these labs to master the bridge between theoretical math and practical Python implementation.

---

###  Core Concepts Mastered

* **Supervised Machine Learning:** * Implementation of Linear and Logistic Regression using **Gradient Descent** and **Vectorization**.
    * Application of **Regularization (L1/L2)** to ensure model generalization and prevent overfitting.

* **Advanced Learning Algorithms:** * Design and training of **Neural Networks** including backpropagation and activation function optimization.
    * Development of **Tree Ensembles**, specifically focusing on the **Random Forest** algorithm for robust classification.



###  Advanced Performance Evaluation & Metrics

I don't just optimize for Accuracy; I evaluate model robustness using the specific metrics required for the problem type:

#### 1. Classification Metrics (For Skewed & Imbalanced Data)
* **Precision & Recall:** Balancing the trade-off between "quality" and "quantity" of predictions.
* **$F_1$ Score:** Utilizing the harmonic mean to evaluate models where False Positives and False Negatives carry high costs.
  $$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$
* **Confusion Matrix:** Analyzing True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN) to identify specific model biases.



#### 2. Regression Metrics (For Continuous Values)
* **Mean Squared Error (MSE):** Measuring the average squared difference between estimated values and the actual value.
  $$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{\mathbf{w}, b}(\mathbf{x}^{(i)}) - y^{(i)})^2$$
* **Mean Absolute Error (MAE):** Providing a linear score where all individual differences are weighted equally.

#### 3. Neural Network & Tree Optimization
* **Cross-Entropy Loss:** The primary cost function I use for binary and multi-class classification in Neural Networks.
* **Entropy & Information Gain:** The mathematical basis I use for splitting nodes in **Decision Trees** and **Random Forests**.
*
---

### 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow/Keras
