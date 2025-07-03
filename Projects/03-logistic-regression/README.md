# 🧠 Breast Cancer Classification using Logistic Regression (From Scratch & Scikit-learn)

Welcome to this beginner-friendly yet insightful machine learning project where we built a **Logistic Regression model entirely from scratch** and also using **scikit-learn**. The focus of this project is on classifying breast cancer tumors as **malignant (M)** or **benign (B)** using numerical features extracted from medical imaging data.

---

## 📁 Project Structure

| File/Folder                | Description                                                                    |
| -------------------------- | ------------------------------------------------------------------------------ |
| `data.csv`                 | Preprocessed breast cancer dataset used for training and testing               |
| `data_visualization.ipynb` | Exploratory Data Analysis using seaborn and matplotlib                         |
| `output.png`               | Pairplot visualization from EDA for feature correlation insight                |
| `scratch_logistic.py`      | Custom implementation of logistic regression with L2 regularization            |
| `notebook.ipynb`           | Step-by-step training, evaluation, and testing pipeline with the scratch model |
| `sklearn_model/`           | Contains the scikit-learn implementation of the logistic regression model.     |


---

## 🚀 What We Did

### 1. **Data Preparation & Normalization**

We began with a dataset containing 569 samples and 30 numerical features including radius, texture, and smoothness. The binary target variable — **malignant (M)** or **benign (B)** — was encoded as 1 and 0, respectively. Feature normalization was performed to ensure a smoother and faster optimization process.

### 2. **Exploratory Data Analysis (EDA)**

In `data_visualization.ipynb`, we conducted a visual examination of the data:

- Generated pairplots using seaborn to explore feature relationships
- Visualized feature correlation and variance
- Identified outliers and key patterns

The pairplot image is saved as `output.png` for quick reference.

### 3. **Stratified Train-Test Split**

To maintain class balance across training and testing sets, we implemented a **custom stratified split** without relying on sklearn. This ensured that both datasets preserved the original ratio of malignant and benign samples, helping to avoid bias and improving reliability.

### 4. **Building Logistic Regression from Scratch**

The centerpiece of this project is `scratch_logistic.py`, where we:

- Implemented the **sigmoid activation**, **binary cross-entropy loss**, and **gradient descent optimization** manually
- Developed core methods: `predict()`, `evaluate()`, `train()`, and `plot_loss()`
- Avoided all high-level machine learning libraries for model training

### 5. **L2 Regularization**

To enhance generalization and combat overfitting:

- We added an **L2 regularization term** to both the loss and gradient calculations
- Introduced a hyperparameter `l2_lambda` to control regularization strength

### 6. **Early Stopping** *(Bonus)*

We implemented early stopping logic to monitor validation loss during training:

- Training stops when the validation loss does not improve for a set number of epochs
- Prevents unnecessary computation and overfitting

### 7. **Model Comparison**

| Model | Accuracy |
| --- | --- |
| Scratch Logistic Regression | 95.58% |
| Scikit-learn Logistic Regression | 97.37% |

Our from-scratch logistic regression model achieved a **95.58% accuracy** on the test set — a remarkable result comparable to that of scikit-learn’s implementation which achieved **97.37% accuracy**.

---

## 📚 Key Concepts Reinforced

- Data preprocessing and normalization
- Manual implementation of gradient descent
- Binary classification using logistic regression
- L2 regularization to improve generalization
- Early stopping for efficient training
- Debugging and structuring reusable ML code

---

## 🧱 Challenges Faced & How We Solved Them

| Challenge                                | Solution                                              |
| ---------------------------------------- | ----------------------------------------------------- |
| Shape mismatches and broadcasting errors | Careful reshaping and inspection of matrix dimensions |
| Manual dataset splitting                 | Implemented a custom stratified split method          |
| Weight updates not reflected             | Ensured weights were stored as class-level attributes |
| No clear stopping point during training  | Added validation-based early stopping logic           |

---

## 🌟 Future Scope

- Add a **confusion matrix** and **classification report** for deeper evaluation
- Build an interactive **Streamlit UI** for real-time predictions
- Extend the implementation to **multi-class classification** problems
- Compare results with advanced models like **XGBoost**, **Random Forest**, etc.
- Deploy the model as a **REST API** for production use

---

## 🔧 How to Use This Project

1. **Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/ML-fundamentals.git
```

2. **Run the code:**

   - Open `notebook.ipynb` to follow the process step by step
   - Or execute `scratch_logistic.py` to train and evaluate the model directly

3. **Visualize the training progress:**

   - Use `plot_loss()` to see how the model’s loss changes over time

---

## 🙌 Final Thoughts

This project wasn’t about using fancy tools — it was about **truly understanding** how machine learning works at its core. By implementing everything ourselves, we built not just a model, but also a strong foundation in ML theory and practice.

If you're someone starting your ML journey or aiming to demystify what goes on behind the scenes in a model like logistic regression — this is a fantastic project to learn from.

> Made with ❤️, Python, and a deep love for learning.

