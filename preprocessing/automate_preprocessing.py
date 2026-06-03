import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def run_preprocessing(input_filepath, output_dir='diabetes_preprocessing'):    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder '{output_dir}' sukses dibuat.")    
    
    if not os.path.exists(input_filepath):
        print(f"Error: File '{input_filepath}' tidak ditemukan. Silakan periksa kembali path file Anda.")
        sys.exit(1)
        
    print(f"Membaca data dari {input_filepath}...")
    df = pd.read_csv(input_filepath)    
    
    if 'Outcome' not in df.columns:
        print("Error: Kolom target 'Outcome' tidak ditemukan di dalam dataset.")
        sys.exit(1)
    
   
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    
    cols_with_zero = [col for col in cols_with_zero if col in df.columns]
    
    if cols_with_zero:
        print("Mengganti nilai 0 dengan NaN pada kolom medis...")
        for col in cols_with_zero:
            df[col] = df[col].replace(0, np.nan)
            
        
        print("Melakukan imputasi nilai kosong menggunakan strategi median...")
        imputer = SimpleImputer(strategy="median")
        df[cols_with_zero] = imputer.fit_transform(df[cols_with_zero])
    
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    
    print("Membagi dataset menjadi Train Set dan Test Set (rasio 75:25)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.25, 
        random_state=42, 
        shuffle=True, 
        stratify=y
    )
    
    
    print("Melakukan standarisasi skala fitur menggunakan StandardScaler...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    
    train_data = pd.concat([X_train_final, y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test_final, y_test.reset_index(drop=True)], axis=1)
    
    
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print("\nProses selesai!")
    print(f"Data Train berhasil disimpan di: {train_path} (Jumlah baris: {train_data.shape[0]})")
    print(f"Data Test berhasil disimpan di: {test_path} (Jumlah baris: {test_data.shape[0]})")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Automasi preprocessing dataset Diabetes.")    
   
    parser.add_argument('input_file', type=str, help='Nama file atau path dataset CSV yang akan diproses.')    
    
    parser.add_argument('--output_dir', type=str, default='diabetes_preprocessing', 
                        help='Nama folder untuk menyimpan hasil train.csv dan test.csv (default: diabetes_preprocessing).')
    
    args = parser.parse_args()
    
    run_preprocessing(input_filepath=args.input_file, output_dir=args.output_dir)
