from datetime import datetime
import pandas as pd
import os
import time
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.decorators import task

from calculate_microloan_fraud_score import calculate_fraud_score

AIRFLOW_DATA_DIR = 'airflow/dags/data'
INPUT_CSV = os.path.join(AIRFLOW_DATA_DIR, 'ppp-full.csv')
OUTPUT_CSV = os.path.join(AIRFLOW_DATA_DIR, 'ppp-full-filtered.csv')
CHUNKS_DIR = os.path.join(AIRFLOW_DATA_DIR, 'chunks')
FILTERED_DIR = os.path.join(AIRFLOW_DATA_DIR, 'filtered_chunks')
PROCESSED_CSV = os.path.join(AIRFLOW_DATA_DIR, 'output_preprocessed.csv')
SCORED_CSV = os.path.join(AIRFLOW_DATA_DIR, 'output_fraud_scored.csv')

with DAG(
    dag_id='preprocessing',
    catchup=False,
    tags=['preprocessing','chunked'],
) as dag:

    @task
    def split_data(input_path: str, chunks_dir: str, chunk_size: int = 100_000) -> list:
        os.makedirs(chunks_dir, exist_ok=True)
        # Check for existing chunk files
        existing = sorted(
            f for f in os.listdir(chunks_dir)
            if f.startswith('chunk_') and f.endswith('.csv')
        )
        if existing:
            return [os.path.join(chunks_dir, fname) for fname in existing]

        # Otherwise, read and split
        chunk_paths = []
        for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
            chunk_path = os.path.join(chunks_dir, f'chunk_{i}.csv')
            chunk.to_csv(chunk_path, index=False)
            chunk_paths.append(chunk_path)
        return chunk_paths

    @task
    def filter_data(chunk_path: str, filtered_dir: str) -> str:
        time.sleep(2) 
        os.makedirs(filtered_dir, exist_ok=True)
        df = pd.read_csv(chunk_path)
        filtered = df[(df['InitialApprovalAmount'] >= 10_000) & (df['InitialApprovalAmount'] <= 11_000)]
        if filtered.empty:
            return ''
        output_path = os.path.join(filtered_dir, os.path.basename(chunk_path))
        filtered.to_csv(output_path, index=False)
        return output_path

    @task
    def merge_data(filtered_paths: list, output_csv: str) -> str:
        # Clean up existing output
        if os.path.exists(output_csv):
            os.remove(output_csv)

        first = True
        processed_count = 0
        for path in filtered_paths:
            if not path:
                continue
            df = pd.read_csv(path)
            df.to_csv(
                output_csv,
                mode='w' if first else 'a',
                header=first,
                index=False
            )
            first = False
            processed_count += 1

        return output_csv

    @task
    def preprocess_data(input_csv: str, output_csv: str) -> str:
        df = pd.read_csv(input_csv)
        # Lowercase and fillna for text fields
        df['BorrowerName_lower'] = df['BorrowerName'].str.lower().fillna('')
        df['BorrowerAddress_lower'] = df['BorrowerAddress'].str.lower().fillna('')
        df['BusinessAgeDescription_lower'] = df['BusinessAgeDescription'].str.lower().fillna('')
        # Numeric conversions and fillna
        df['JobsReported'] = df['JobsReported'].fillna(0).astype(float)
        df['InitialApprovalAmount'] = df['InitialApprovalAmount'].astype(float)
        df['Latitude'] = df['Latitude'].fillna(0).astype(float)
        df['Longitude'] = df['Longitude'].fillna(0).astype(float)
        df['SBAOfficeCode'] = df['SBAOfficeCode'].fillna('')
        # Write processed file
        df.to_csv(output_csv, index=False)
        return f'Preprocessed data saved to {output_csv}'
    
    @task
    def filter_fraud_data(input_csv: str, output_csv: str) -> str:
        df = pd.read_csv(input_csv)
        df = df.iloc[:, :3]
        df = df[df.iloc[:, 2] >= 140]
        df = df.sort_values(by=df.columns[2])
        df.to_csv(input_csv, index=False)
        return f'Final data saved to {output_csv}'
            
    chunk_paths = split_data(INPUT_CSV, CHUNKS_DIR, 100_000)
    filtered_paths = filter_data.partial(filtered_dir=FILTERED_DIR).expand(
        chunk_path=chunk_paths
    )
    output_path = merge_data(filtered_paths, OUTPUT_CSV)
    preprocess_task = preprocess_data(output_path, PROCESSED_CSV)

    calculate_score_task = PythonOperator(
        task_id='calculate_fraud_score',
        python_callable=calculate_fraud_score,
        op_kwargs={
            'input_file': PROCESSED_CSV,
            'output_file': SCORED_CSV
        }
    )

    preprocess_task >> calculate_score_task

    filter_fraud_data_task = filter_fraud_data(SCORED_CSV, "final_output.txt")

    calculate_score_task >> filter_fraud_data_task
