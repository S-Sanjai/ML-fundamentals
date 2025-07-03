# Logistic Regression for Classification

A comprehensive implementation of logistic regression from scratch for binary and multiclass classification problems.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements logistic regression from scratch using Python and NumPy. The implementation includes both binary and multiclass classification capabilities, demonstrating the mathematical foundations and practical applications of logistic regression in machine learning.

## ✨ Features

- **From-scratch implementation** of logistic regression
- **Binary classification** support
- **Multiclass classification** using One-vs-Rest strategy
- **Gradient descent optimization** with customizable learning rates
- **Regularization options** (L1, L2, and Elastic Net)
- **Comprehensive evaluation metrics** (accuracy, precision, recall, F1-score, ROC-AUC)
- **Data visualization** of decision boundaries and learning curves
- **Cross-validation** for model validation
- **Feature scaling and preprocessing** utilities

## 📊 Dataset

This project uses [specify your dataset here, e.g.]:
- **Primary Dataset**: [Dataset name and source]
- **Features**: [Number] features including [brief description]
- **Target**: [Description of target variable]
- **Size**: [Number of samples] samples

### Data Preprocessing
- Handle missing values
- Feature scaling/normalization
- Categorical variable encoding
- Train-test split with stratification

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/S-Sanjai/ML-fundamentals.git
cd ML-fundamentals/Projects/02-logistic-regression-for-classification
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 💻 Usage

### Quick Start
```python
from logistic_regression import LogisticRegression
from data_utils import load_and_preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/dataset.csv')

# Initialize and train model
model = LogisticRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

### Running the Complete Pipeline
```bash
# Run the main script
python main.py

# Run with custom parameters
python main.py --learning_rate 0.001 --max_iter 2000 --regularization l2
```

### Jupyter Notebook
Open and run the provided notebook for interactive exploration:
```bash
jupyter notebook logistic_regression_analysis.ipynb
```

## 🔧 Implementation Details

### Mathematical Foundation
The logistic regression model is based on the following key concepts:

1. **Sigmoid Function**: 
   ```
   σ(z) = 1 / (1 + e^(-z))
   ```

2. **Cost Function** (Log-likelihood):
   ```
   J(θ) = -1/m * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
   ```

3. **Gradient Descent Update**:
   ```
   θ := θ - α * (1/m) * X^T * (h(x) - y)
   ```

### Key Components

#### LogisticRegression Class
- `fit(X, y)`: Train the model using gradient descent
- `predict(X)`: Make binary predictions
- `predict_proba(X)`: Return prediction probabilities
- `score(X, y)`: Calculate accuracy score

#### Regularization
- **L1 Regularization (Lasso)**: Promotes sparsity
- **L2 Regularization (Ridge)**: Prevents overfitting
- **Elastic Net**: Combination of L1 and L2

## 📈 Results

### Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | [Your accuracy]% |
| Precision | [Your precision] |
| Recall | [Your recall] |
| F1-Score | [Your F1-score] |
| ROC-AUC | [Your ROC-AUC] |

### Visualizations
The project includes several visualization components:
- Decision boundary plots
- Learning curves
- Feature importance analysis
- ROC curves
- Confusion matrices

## 📁 Project Structure
```
02-logistic-regression-for-classification/
├── data/
│   ├── raw/                    # Raw datasets
│   └── processed/              # Preprocessed data
├── src/
│   ├── logistic_regression.py  # Main implementation
│   ├── data_utils.py          # Data preprocessing utilities
│   ├── visualization.py       # Plotting functions
│   └── evaluation.py          # Evaluation metrics
├── notebooks/
│   └── logistic_regression_analysis.ipynb
├── tests/
│   ├── test_logistic_regression.py
│   └── test_data_utils.py
├── results/
│   ├── plots/                 # Generated visualizations
│   └── models/                # Saved model parameters
├── requirements.txt
├── main.py                    # Main execution script
└── README.md
```

## 📦 Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **scikit-learn**: Evaluation metrics and data utilities
- **jupyter**: Interactive notebooks

Install all dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [Logistic Regression - Andrew Ng's Course](https://www.coursera.org/learn/machine-learning)
- [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
- [Pattern Recognition and Machine Learning - Bishop](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)

## 👤 Author

**S. Sanjai**
- GitHub: [@S-Sanjai](https://github.com/S-Sanjai)
- Project Link: [ML-fundamentals](https://github.com/S-Sanjai/ML-fundamentals)

---
⭐ If you found this project helpful, please give it a star!