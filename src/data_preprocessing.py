import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'telco_churn.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_data.csv')

if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
    os.makedirs(os.path.dirname(OUTPUT_PATH))

if os.path.exists(INPUT_PATH):
    df = pd.read_csv(INPUT_PATH)
    df.columns = df.columns.str.strip()
    
    if 'Churn Label' in df.columns:
        df['TARGET_CHURN'] = df['Churn Label'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    
    cols_to_ignore = [
        'Customer ID', 'Churn Label', 'City', 'Zip Code', 'Latitude', 'Longitude', 
        'Churn Category', 'Churn Reason', 'Customer Status', 'Churn Score', 'CLTV',
        'Country', 'State', 'Population', 'Quarter', 'Satisfaction Score'
    ]
    df = df.drop(columns=[c for c in cols_to_ignore if c in df.columns])
    
    if 'Total Charges' in df.columns:
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)

    if 'Monthly Charge' in df.columns:
        df['Monthly Charge'] = pd.to_numeric(df['Monthly Charge'], errors='coerce').fillna(0)

    df = pd.get_dummies(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Preprocessing Complete.")