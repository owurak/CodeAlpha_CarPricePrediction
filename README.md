# CodeAlpha_CarPricePrediction
Machine Learning model to predict car prices using Random Forest Regression

# 🚗 Car Price Prediction using Machine Learning

This project predicts the selling price of used cars using machine learning techniques. It was developed as part of the CodeAlpha Data Science Internship.

---

## 📊 Dataset
The dataset used is the CarDekho dataset from Kaggle, which contains information about used cars such as:
- Car name
- Year of manufacture
- Selling price
- Present price
- Kilometers driven
- Fuel type
- Seller type
- Transmission
- Number of previous owners

---

## 🧠 Objective
To build a regression model that can accurately predict the selling price of a used car based on its features.

---

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

---

## 🤖 Machine Learning Model
- Random Forest Regressor

---

## 🧹 Data Preprocessing
- Removed unnecessary columns (Car Name)
- Converted Year into Car Age
- Handled categorical variables using One-Hot Encoding
- Split dataset into training and testing sets

---

## 📈 Model Evaluation
- R² Score: ~0.95
- Mean Absolute Error (MAE): ~0.6

The model shows strong performance in predicting car prices.

---

## 🚀 How to Run the Project

3. Run the script:

1. Clone this repository
2. Install required libraries:
3. 2. Install required libraries:

```bash
pip install numpy pandas scikit-learn
python car_price_prediction.py
