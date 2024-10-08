import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def handle_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    # Filter out outliers
    df_filtered = df[~((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))).any(axis=1)]
    
    # Alternative Approach:
    # lower = Q1 - 1.5*IQR
    # upper = Q3 + 1.5*IQR
    # df_filtered = df.clip(lower=lower, upper=upper, axis=1)
    
    return df_filtered

def main():
    # Load Data
    df = pd.read_csv('data/USA_Housing.csv')
    df = df.drop('Address', axis=1)
    # Remove outliers
    # df = handle_outliers(df)
    
    # Separate into features and target variables
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    print(X.columns)
    # Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    # Initialize model
    lr = LinearRegression()
    
    # Train model
    lr.fit(X_train, y_train)
    
    # Predict Model
    y_pred = lr.predict(X_test)
    
    # Evaluate Model
    print(f'MAE: {mean_absolute_error(y_test, y_pred):,.2f}')
    print(f'MSE: {mean_squared_error(y_test, y_pred):,.2f}')
    print(f'R2: {r2_score(y_test, y_pred)*100:.2f}%')
    
    # Save model
    with open('models/model.joblib', 'wb') as f:
        joblib.dump(lr, f)

if __name__ == '__main__':
    main()
