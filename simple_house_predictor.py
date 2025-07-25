import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self, csv_file):
        """Initialize the predictor with the dataset."""
        self.df = pd.read_csv(csv_file)
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def explore_data(self):
        """Explore and analyze the dataset."""
        print("=" * 60)
        print("🏠 HOUSE PRICE PREDICTION - DATASET EXPLORATION")
        print("=" * 60)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumn names: {list(self.df.columns)}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nData types:\n{self.df.dtypes}")
        
        print(f"\nPrice statistics:")
        print(f"  Minimum: ₹{self.df['price'].min():,.2f}")
        print(f"  Maximum: ₹{self.df['price'].max():,.2f}")
        print(f"  Mean: ₹{self.df['price'].mean():,.2f}")
        print(f"  Median: ₹{self.df['price'].median():,.2f}")
        
        print(f"\nFeature summary:")
        for col in self.df.columns:
            if col != 'price':
                if self.df[col].dtype == 'object':
                    print(f"  {col}: {self.df[col].unique()}")
                else:
                    print(f"  {col}: {self.df[col].min()} - {self.df[col].max()}")
        
    def preprocess_data(self):
        """Preprocess the data for machine learning."""
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Identify categorical columns (excluding target)
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Categorical features: {categorical_cols}")
        print(f"Numerical features: {numerical_cols}")
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
            print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Separate features and target
        X = df_processed.drop('price', axis=1)
        y = df_processed['price']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nData split:")
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        print(f"  Features: {X_train.shape[1]}")
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        print(f"  Feature order: {self.feature_names}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple regression models."""
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0, max_iter=2000),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate models with cross-validation
        print("Training models with 5-fold cross-validation...")
        cv_results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Fit the model
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Cross-validation
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                cv_results[name] = cv_scores
                print(f"  ✓ CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            except Exception as e:
                print(f"  ⚠ CV failed: {str(e)}")
                cv_results[name] = np.array([0])
        
        return cv_results
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test data."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION ON TEST DATA")
        print("=" * 60)
        
        results = {}
        predictions = {}
        
        for name, model in self.models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                predictions[name] = y_pred
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2
                }
                
                print(f"\n{name}:")
                print(f"  R² Score: {r2:.4f}")
                print(f"  RMSE: ₹{rmse:,.2f}")
                print(f"  MAE: ₹{mae:,.2f}")
                
                # Show some sample predictions
                print(f"  Sample predictions vs actual:")
                for i in range(min(3, len(y_test))):
                    actual = list(y_test)[i]
                    predicted = y_pred[i]
                    print(f"    Actual: ₹{actual:,.2f}, Predicted: ₹{predicted:,.2f}")
                    
            except Exception as e:
                print(f"  ❌ Evaluation failed: {str(e)}")
                results[name] = {'R²': -999}
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if v['R²'] > -999}
        if valid_results:
            best_model_name = max(valid_results.keys(), key=lambda x: valid_results[x]['R²'])
            print(f"\n🏆 BEST MODEL: {best_model_name}")
            print(f"   R² Score: {valid_results[best_model_name]['R²']:.4f}")
            print(f"   RMSE: ₹{valid_results[best_model_name]['RMSE']:,.2f}")
        else:
            best_model_name = list(self.models.keys())[0]
            print(f"\n⚠ Using default model: {best_model_name}")
        
        return results, predictions, best_model_name
    
    def show_feature_importance(self, best_model_name):
        """Show feature importance if available."""
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        model = self.models[best_model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = list(zip(self.feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Feature importance for {best_model_name}:")
            for feature, importance in feature_importance:
                print(f"  {feature:20s}: {importance:.4f}")
                
        elif hasattr(model, 'coef_'):
            coefficients = abs(model.coef_)
            feature_importance = list(zip(self.feature_names, coefficients))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Feature coefficients (absolute) for {best_model_name}:")
            for feature, coef in feature_importance:
                print(f"  {feature:20s}: {coef:.4f}")
        else:
            print(f"{best_model_name} does not provide feature importance information.")
    
    def make_prediction(self, best_model_name, sample_input=None, show_header=True):
        """Make a prediction with the best model."""
        if show_header:
            print("\n" + "=" * 60)
            print("SAMPLE PREDICTION")
            print("=" * 60)
        
        if sample_input is None:
            # Use a realistic sample
            sample_input = {
                'area': 7500,
                'bedrooms': 4,
                'bathrooms': 2,
                'stories': 2,
                'mainroad': 'yes',
                'guestroom': 'no',
                'basement': 'yes',
                'hotwaterheating': 'no',
                'airconditioning': 'yes',
                'parking': 2,
                'prefarea': 'yes',
                'furnishingstatus': 'furnished'
            }
        
        print("Sample house features:")
        for key, value in sample_input.items():
            print(f"  {key:20s}: {value}")
        
        try:
            # Prepare the input for prediction
            sample_df = pd.DataFrame([sample_input])
            
            # Encode categorical variables
            for col in sample_df.select_dtypes(include=['object']).columns:
                if col in self.label_encoders:
                    sample_df[col] = self.label_encoders[col].transform(sample_df[col])
            
            # Ensure feature order matches training data
            sample_df = sample_df[self.feature_names]
            
            # Scale the input
            sample_scaled = self.scaler.transform(sample_df)
            
            # Make prediction
            best_model = self.models[best_model_name]
            predicted_price = best_model.predict(sample_scaled)[0]
            
            print(f"\n🏠 PREDICTED PRICE: ₹{predicted_price:,.2f}")
            
            # Calculate price per square foot
            price_per_sqft = predicted_price / sample_input['area']
            print(f"📐 Price per sq ft: ₹{price_per_sqft:,.2f}")
            
            return predicted_price
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            return None
    
    def run_complete_analysis(self):
        """Run the complete machine learning pipeline."""
        print("🏠 HOUSE PRICE PREDICTION MODEL")
        print("Using Machine Learning to Predict Real Estate Prices")
        print("=" * 60)
        
        try:
            # Step 1: Explore data
            self.explore_data()
            
            # Step 2: Preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data()
            
            # Step 3: Train models
            cv_scores = self.train_models(X_train, y_train)
            
            # Step 4: Evaluate models
            results, predictions, best_model_name = self.evaluate_models(X_test, y_test)
            
            # Step 5: Show feature importance
            self.show_feature_importance(best_model_name)
            
            # Step 6: Make sample prediction
            self.make_prediction(best_model_name)
            
            print("\n" + "=" * 60)
            print("✅ ANALYSIS COMPLETE!")
            print("=" * 60)
            print("📊 The model has been trained and evaluated successfully.")
            print(f"🎯 Best performing model: {best_model_name}")
            print("💡 You can now use this model to predict house prices!")
            
            return best_model_name, results
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return None, None

# Function to get user input for house details
def get_user_house_details():
    """Prompt user to input their house details interactively."""
    print("\n" + "=" * 60)
    print("🏠 ENTER YOUR HOUSE DETAILS FOR PRICE PREDICTION")
    print("=" * 60)
    print("Please provide the following information about your house:")
    print()
    
    house_details = {}
    
    # Get area
    while True:
        try:
            area = int(input("📏 House area (in square feet, range 1650-16200): "))
            if 1000 <= area <= 20000:  # Allow slightly wider range than dataset
                house_details['area'] = area
                break
            else:
                print("   ⚠️  Please enter a valid area between 1,000 and 20,000 sq ft")
        except ValueError:
            print("   ⚠️  Please enter a valid number for area")
    
    # Get bedrooms
    while True:
        try:
            bedrooms = int(input("🛏️  Number of bedrooms (1-6): "))
            if 1 <= bedrooms <= 6:
                house_details['bedrooms'] = bedrooms
                break
            else:
                print("   ⚠️  Please enter between 1 and 6 bedrooms")
        except ValueError:
            print("   ⚠️  Please enter a valid number for bedrooms")
    
    # Get bathrooms
    while True:
        try:
            bathrooms = int(input("🚿 Number of bathrooms (1-4): "))
            if 1 <= bathrooms <= 4:
                house_details['bathrooms'] = bathrooms
                break
            else:
                print("   ⚠️  Please enter between 1 and 4 bathrooms")
        except ValueError:
            print("   ⚠️  Please enter a valid number for bathrooms")
    
    # Get stories
    while True:
        try:
            stories = int(input("🏢 Number of stories/floors (1-4): "))
            if 1 <= stories <= 4:
                house_details['stories'] = stories
                break
            else:
                print("   ⚠️  Please enter between 1 and 4 stories")
        except ValueError:
            print("   ⚠️  Please enter a valid number for stories")
    
    # Get parking
    while True:
        try:
            parking = int(input("🚗 Number of parking spaces (0-3): "))
            if 0 <= parking <= 3:
                house_details['parking'] = parking
                break
            else:
                print("   ⚠️  Please enter between 0 and 3 parking spaces")
        except ValueError:
            print("   ⚠️  Please enter a valid number for parking spaces")
    
    # Get yes/no features
    yes_no_features = {
        'mainroad': '🛣️  Is the house on main road? (yes/no): ',
        'guestroom': '🏠 Does the house have a guest room? (yes/no): ',
        'basement': '🏚️  Does the house have a basement? (yes/no): ',
        'hotwaterheating': '🔥 Does the house have hot water heating? (yes/no): ',
        'airconditioning': '❄️  Does the house have air conditioning? (yes/no): ',
        'prefarea': '⭐ Is the house in a preferred area? (yes/no): '
    }
    
    for feature, prompt in yes_no_features.items():
        while True:
            response = input(prompt).lower().strip()
            if response in ['yes', 'y', 'no', 'n']:
                house_details[feature] = 'yes' if response in ['yes', 'y'] else 'no'
                break
            else:
                print("   ⚠️  Please enter 'yes' or 'no'")
    
    # Get furnishing status
    print("\n🪑 Furnishing status:")
    print("   1. Furnished")
    print("   2. Semi-furnished") 
    print("   3. Unfurnished")
    
    while True:
        try:
            furnish_choice = int(input("Select furnishing status (1-3): "))
            if furnish_choice == 1:
                house_details['furnishingstatus'] = 'furnished'
                break
            elif furnish_choice == 2:
                house_details['furnishingstatus'] = 'semi-furnished'
                break
            elif furnish_choice == 3:
                house_details['furnishingstatus'] = 'unfurnished'
                break
            else:
                print("   ⚠️  Please enter 1, 2, or 3")
        except ValueError:
            print("   ⚠️  Please enter a valid number (1, 2, or 3)")
    
    return house_details

def make_user_prediction(predictor, best_model_name):
    """Get user input and make a prediction."""
    print("\n" + "=" * 60)
    print("🎯 INTERACTIVE HOUSE PRICE PREDICTION")
    print("=" * 60)
    
    # Get user input
    user_house = get_user_house_details()
    
    # Display entered details
    print("\n" + "=" * 60)
    print("📋 YOUR HOUSE DETAILS SUMMARY")
    print("=" * 60)
    for key, value in user_house.items():
        print(f"  {key:20s}: {value}")
    
    # Ask for confirmation
    print("\n🤔 Do you want to proceed with price prediction? (yes/no): ", end="")
    confirm = input().lower().strip()
    
    if confirm in ['yes', 'y']:
        # Make prediction
        predicted_price = predictor.make_prediction(best_model_name, user_house, show_header=False)
        
        if predicted_price:
            print(f"\n💰 ESTIMATED MARKET VALUE: ₹{predicted_price:,.2f}")
            print(f"📊 Price Range: ₹{predicted_price*0.9:,.2f} - ₹{predicted_price*1.1:,.2f}")
            print(f"📐 Price per sq ft: ₹{predicted_price/user_house['area']:,.2f}")
            
            # Additional insights
            print(f"\n💡 INSIGHTS:")
            if user_house['area'] > 7000:
                print("   🏰 Large house - Premium property")
            elif user_house['area'] > 4000:
                print("   🏠 Medium-sized house - Good family home")
            else:
                print("   🏡 Compact house - Suitable for small family")
                
            if user_house['airconditioning'] == 'yes':
                print("   ❄️  AC adds significant value to your property")
            if user_house['parking'] >= 2:
                print("   🚗 Good parking availability increases property value")
            if user_house['prefarea'] == 'yes':
                print("   ⭐ Premium location adds extra value")
        
        return predicted_price
    else:
        print("❌ Prediction cancelled.")
        return None

# Function to make custom predictions (for demo purposes)
def make_demo_prediction(predictor, best_model_name):
    """Make a demo prediction with sample data."""
    print("\n" + "=" * 60)
    print("📊 DEMO PREDICTION")
    print("=" * 60)
    
    # Example custom house
    demo_house = {
        'area': 6000,
        'bedrooms': 3,
        'bathrooms': 2,
        'stories': 1,
        'mainroad': 'yes',
        'guestroom': 'yes',
        'basement': 'no',
        'hotwaterheating': 'no',
        'airconditioning': 'yes',
        'parking': 1,
        'prefarea': 'no',
        'furnishingstatus': 'semi-furnished'
    }
    
    print("Demo house features:")
    predicted_price = predictor.make_prediction(best_model_name, demo_house)
    
    return predicted_price

# Main execution
if __name__ == "__main__":
    print("Starting House Price Prediction Analysis...")
    
    # Create predictor instance
    predictor = HousePricePredictor('Housing.csv')
    
    # Run complete analysis
    best_model, results = predictor.run_complete_analysis()
    
    if best_model and results:
        print(f"\n🎉 Success! Your {best_model} model is ready to predict house prices!")
        
        # Interactive menu for user
        while True:
            print("\n" + "=" * 60)
            print("🏠 HOUSE PRICE PREDICTION MENU")
            print("=" * 60)
            print("Choose an option:")
            print("  1. 🎯 Enter YOUR house details for price prediction")
            print("  2. 📊 See a demo prediction")
            print("  3. 🚪 Exit")
            print()
            
            try:
                choice = input("Enter your choice (1-3): ").strip()
                
                if choice == '1':
                    # User enters their house details
                    user_price = make_user_prediction(predictor, best_model)
                    if user_price:
                        # Ask if user wants to predict another house
                        continue_choice = input("\n🔄 Would you like to predict another house price? (yes/no): ").lower().strip()
                        if continue_choice not in ['yes', 'y']:
                            break
                    
                elif choice == '2':
                    # Show demo prediction
                    demo_price = make_demo_prediction(predictor, best_model)
                    
                elif choice == '3':
                    print("\n👋 Thank you for using the House Price Predictor!")
                    print("🏠 Happy house hunting! 🎉")
                    break
                    
                else:
                    print("   ⚠️  Please enter 1, 2, or 3")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using the House Price Predictor!")
                break
            except Exception as e:
                print(f"   ❌ An error occurred: {str(e)}")
                print("   Please try again.")
    else:
        print("❌ Analysis failed. Please check your data and try again.") 