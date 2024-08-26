import pandas as pd

# Load Data
df = pd.read_csv('data/USA_Housing.csv')
df = df.drop('Address', axis=1)
print(df.columns)

def handle_outlers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    # Filter out outliers
    df_filtered = df[~((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.3*IQR))).any(axis=1)]
    return df_filtered