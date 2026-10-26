from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.validation import validate_data

def prepare_data():
    
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / 'data' / 'raw'/ 'UCI_Credit_Card.csv'
    
    
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(data_path)
    target = 'default.payment.next.month'
    
    
    df = pd.read_csv(data_path)
    validate_data(df)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])   
    
    train_df.to_csv(output_dir / 'train_data.csv', index=False)
    test_df.to_csv(output_dir / 'test_data.csv', index=False)
    
    print(f"Train and test datasets have been saved to {output_dir}.")
    
if __name__ == "__main__":
    prepare_data()
    