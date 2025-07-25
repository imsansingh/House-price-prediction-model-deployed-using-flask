from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import os

app = Flask(__name__)

class HousePriceModel:
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def load_and_train_model(self, csv_file='Housing.csv'):
        """Load data and train the model."""
        try:
            # Load data
            df = pd.read_csv(csv_file)
            
            # Preprocess data
            df_processed = df.copy()
            
            # Encode categorical variables
            categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
            
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
            
            # Separate features and target
            X = df_processed.drop('price', axis=1)
            y = df_processed['price']
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train Gradient Boosting model (best performer)
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            self.is_trained = True
            print("✅ Model trained successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {str(e)}")
            return False
    
    def predict_price(self, house_data):
        """Predict house price from input data."""
        if not self.is_trained:
            return None, "Model not trained"
        
        try:
            # Convert to DataFrame
            input_df = pd.DataFrame([house_data])
            
            # Encode categorical variables
            for col in input_df.select_dtypes(include=['object']).columns:
                if col in self.label_encoders:
                    input_df[col] = self.label_encoders[col].transform(input_df[col])
            
            # Ensure feature order matches training data
            input_df = input_df[self.feature_names]
            
            # Scale the input
            input_scaled = self.scaler.transform(input_df)
            
            # Make prediction
            predicted_price = self.model.predict(input_scaled)[0]
            
            return predicted_price, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

# Initialize model
price_model = HousePriceModel()

@app.route('/')
def index():
    """Home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request."""
    try:
        # Get form data
        house_data = {
            'area': int(request.form['area']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'stories': int(request.form['stories']),
            'mainroad': request.form['mainroad'],
            'guestroom': request.form['guestroom'],
            'basement': request.form['basement'],
            'hotwaterheating': request.form['hotwaterheating'],
            'airconditioning': request.form['airconditioning'],
            'parking': int(request.form['parking']),
            'prefarea': request.form['prefarea'],
            'furnishingstatus': request.form['furnishingstatus']
        }
        
        # Validate inputs
        validation_errors = validate_inputs(house_data)
        if validation_errors:
            return render_template('index.html', 
                                 error=validation_errors, 
                                 form_data=house_data)
        
        # Make prediction
        predicted_price, error = price_model.predict_price(house_data)
        
        if error:
            return render_template('index.html', 
                                 error=error, 
                                 form_data=house_data)
        
        # Calculate additional metrics
        price_per_sqft = predicted_price / house_data['area']
        price_range_low = predicted_price * 0.9
        price_range_high = predicted_price * 1.1
        
        # Generate insights
        insights = generate_insights(house_data, predicted_price)
        
        return render_template('result.html',
                             house_data=house_data,
                             predicted_price=predicted_price,
                             price_per_sqft=price_per_sqft,
                             price_range_low=price_range_low,
                             price_range_high=price_range_high,
                             insights=insights)
        
    except Exception as e:
        return render_template('index.html', 
                             error=f"An error occurred: {str(e)}", 
                             form_data=request.form.to_dict())

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                          'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
                          'parking', 'prefarea', 'furnishingstatus']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Make prediction
        predicted_price, error = price_model.predict_price(data)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'predicted_price': predicted_price,
            'price_per_sqft': predicted_price / data['area'],
            'price_range': {
                'low': predicted_price * 0.9,
                'high': predicted_price * 1.1
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def validate_inputs(house_data):
    """Validate user inputs."""
    errors = []
    
    # Validate area
    if house_data['area'] < 1000 or house_data['area'] > 20000:
        errors.append("Area must be between 1,000 and 20,000 sq ft")
    
    # Validate bedrooms
    if house_data['bedrooms'] < 1 or house_data['bedrooms'] > 6:
        errors.append("Bedrooms must be between 1 and 6")
    
    # Validate bathrooms
    if house_data['bathrooms'] < 1 or house_data['bathrooms'] > 4:
        errors.append("Bathrooms must be between 1 and 4")
    
    # Validate stories
    if house_data['stories'] < 1 or house_data['stories'] > 4:
        errors.append("Stories must be between 1 and 4")
    
    # Validate parking
    if house_data['parking'] < 0 or house_data['parking'] > 3:
        errors.append("Parking spaces must be between 0 and 3")
    
    return errors

def generate_insights(house_data, predicted_price):
    """Generate insights about the house."""
    insights = []
    
    # Size insights
    if house_data['area'] > 7000:
        insights.append("🏰 Large house - Premium property")
    elif house_data['area'] > 4000:
        insights.append("🏠 Medium-sized house - Good family home")
    else:
        insights.append("🏡 Compact house - Suitable for small family")
    
    # Feature insights
    if house_data['airconditioning'] == 'yes':
        insights.append("❄️ Air conditioning adds significant value to your property")
    
    if house_data['parking'] >= 2:
        insights.append("🚗 Good parking availability increases property value")
    
    if house_data['prefarea'] == 'yes':
        insights.append("⭐ Premium location adds extra value")
    
    if house_data['basement'] == 'yes':
        insights.append("🏚️ Basement provides additional storage/utility space")
    
    if house_data['guestroom'] == 'yes':
        insights.append("🏠 Guest room adds flexibility and value")
    
    # Price insights
    price_per_sqft = predicted_price / house_data['area']
    if price_per_sqft > 1500:
        insights.append("💰 High price per sq ft - Premium property")
    elif price_per_sqft > 1000:
        insights.append("💰 Moderate price per sq ft - Good value")
    else:
        insights.append("💰 Affordable price per sq ft - Budget-friendly")
    
    return insights

if __name__ == '__main__':
    print("🏠 Starting House Price Prediction Web App...")
    
    # Train the model
    if price_model.load_and_train_model():
        print("🚀 Model ready! Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to train model. Please check Housing.csv file.") 