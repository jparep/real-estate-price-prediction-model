import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def handle_outlers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    # Filter out outliers
    df_filtered = df[~((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.3*IQR))).any(axis=1)]
    
    #ALternative Appriach
    """
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df_filtered = df.clip(lower=lower, upper=upper, axis=1)
    """
    return df_filtered


def main():
    # Load Data
    df = pd.read_csv('data/USA_Housing.csv')
    df = df.drop('Address', axis=1)
    
    # Remove outliers
    df = handle_outlers(df)
    
    # Separate into features and target variables
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # SPlit data into train and test set
    X_train, X_test, y_train, y_tet =train_test_split(X,y, test_size=0.2, random_state=12)
    
    # Initalize model
    lr = LinearRegression()
    
    # Train model
    lr.fit(X_train, y_train)
    
    # Predict Model
    
    