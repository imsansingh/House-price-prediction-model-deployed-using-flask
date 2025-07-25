# 🏠 House Price Prediction Model

A machine learning regression model to predict house prices using the Housing.csv dataset.

## 📊 Model Performance Results

### Best Model: **Gradient Boosting Regressor**
- **R² Score**: 0.6647 (66.47% of price variation explained)
- **RMSE**: ₹1,301,871.87
- **MAE**: ₹964,058.87

### All Models Comparison:
| Model | R² Score | RMSE (₹) | MAE (₹) |
|-------|----------|----------|---------|
| **Gradient Boosting** | **0.6647** | **1,301,871** | **964,058** |
| Linear Regression | 0.6495 | 1,331,071 | 979,679 |
| Lasso Regression | 0.6495 | 1,331,072 | 979,680 |
| Ridge Regression | 0.6494 | 1,331,290 | 979,549 |
| Random Forest | 0.6118 | 1,400,765 | 1,026,699 |

## 🎯 Dataset Overview

- **Total Records**: 545 houses
- **Features**: 12 (area, bedrooms, bathrooms, etc.)
- **Price Range**: ₹1.75M - ₹13.3M
- **Average Price**: ₹4.77M
- **No Missing Values**: Clean dataset ready for ML

### Features Used:
- **area**: House area in square feet
- **bedrooms**: Number of bedrooms (1-6)
- **bathrooms**: Number of bathrooms (1-4)
- **stories**: Number of stories (1-4)
- **mainroad**: Access to main road (yes/no)
- **guestroom**: Guest room available (yes/no)
- **basement**: Basement available (yes/no)
- **hotwaterheating**: Hot water heating (yes/no)
- **airconditioning**: Air conditioning (yes/no)
- **parking**: Number of parking spaces (0-3)
- **prefarea**: Located in preferred area (yes/no)
- **furnishingstatus**: furnished/semi-furnished/unfurnished

## 🔍 Feature Importance Analysis

The Gradient Boosting model identified these as the most important features:

| Feature | Importance | Impact |
|---------|------------|--------|
| **area** | 46.10% | 🏠 Most critical factor |
| **bathrooms** | 16.71% | 🚿 Second most important |
| **airconditioning** | 9.10% | ❄️ Significant comfort factor |
| **parking** | 5.15% | 🚗 Parking availability |
| **stories** | 4.57% | 🏢 Building height |
| **bedrooms** | 4.56% | 🛏️ Room count |
| Others | 13.81% | Various smaller factors |

## 🚀 How to Use

### 1. Run the Complete Analysis:
```bash
python simple_house_predictor.py
```

### 2. Make Custom Predictions:
Modify the `custom_house` dictionary in the script with your house features:

```python
custom_house = {
    'area': 6000,               # Square feet
    'bedrooms': 3,              # Number of bedrooms
    'bathrooms': 2,             # Number of bathrooms
    'stories': 1,               # Number of stories
    'mainroad': 'yes',          # yes/no
    'guestroom': 'yes',         # yes/no
    'basement': 'no',           # yes/no
    'hotwaterheating': 'no',    # yes/no
    'airconditioning': 'yes',   # yes/no
    'parking': 1,               # Number of parking spaces
    'prefarea': 'no',           # yes/no
    'furnishingstatus': 'semi-furnished'  # furnished/semi-furnished/unfurnished
}
```

## 📈 Sample Predictions

### Example 1: Premium House
- **Area**: 7,500 sq ft, 4 bed, 2 bath, AC, furnished
- **Predicted Price**: ₹9,432,658 (₹1,257/sq ft)

### Example 2: Mid-Range House  
- **Area**: 6,000 sq ft, 3 bed, 2 bath, AC, semi-furnished
- **Predicted Price**: ₹6,840,249 (₹1,140/sq ft)

## 🛠️ Requirements

Install required packages:
```bash
pip install pandas numpy scikit-learn
```

## 📁 Files

- `Housing.csv` - Original dataset
- `simple_house_predictor.py` - Main ML model script
- `house_price_predictor.py` - Full featured version with visualizations
- `requirements.txt` - Python dependencies

## 🎯 Model Insights

1. **Area is King**: 46% of price variation is explained by house area
2. **Bathrooms Matter**: More bathrooms significantly increase value
3. **Air Conditioning**: Premium feature that adds substantial value
4. **Parking**: Each parking space adds considerable value
5. **Good Performance**: 66.47% accuracy in price prediction

## 🔮 Future Improvements

- Add more location-specific features
- Include neighborhood data
- Consider market trends and timing
- Add more advanced ensemble methods
- Implement feature engineering for better accuracy

## 💡 Business Applications

- **Real Estate Valuation**: Automated property assessment
- **Investment Analysis**: Identify undervalued properties  
- **Market Research**: Understand pricing factors
- **Customer Tools**: Help buyers estimate fair prices

---

🎉 **Your machine learning model is ready to predict house prices with 66.47% accuracy!** 