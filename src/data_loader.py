import pandas as pd
import numpy as np

def load_data():
    columns = ['unit', 'cycle', 'op_setting1', 'op_setting2', 'op_setting3'] + [f'sensor{i}' for i in range(1, 22)]
    
    train_df = pd.read_csv('data/train_FD002.txt', sep=' ', header=None, names=columns, usecols=range(26))
    test_df = pd.read_csv('data/test_FD002.txt', sep=' ', header=None, names=columns, usecols=range(26))
    rul_df = pd.read_csv('data/RUL_FD002.txt', header=None, names=['rul'])
    
    print(f"Train: {train_df.shape} | Test: {test_df.shape} | RUL: {rul_df.shape}")
    
    return train_df, test_df, rul_df

def add_rul_to_train(train_df):
    max_cycles = train_df.groupby('unit')['cycle'].max().reset_index()
    train_df = train_df.merge(max_cycles, on='unit', suffixes=('', '_max'))
    train_df['RUL'] = train_df['cycle_max'] - train_df['cycle']
    train_df.drop('cycle_max', axis=1, inplace=True)
    return train_df

def add_rul_to_test(test_df, rul_df):
    test_last_cycle = test_df.groupby('unit')['cycle'].max().reset_index()
    test_last_cycle.rename(columns={'cycle': 'last_observed_cycle'}, inplace=True)
    test_df = test_df.merge(test_last_cycle, on='unit')
    
    rul_df['unit'] = rul_df.index + 1
    
    test_last_rows = test_df[test_df['cycle'] == test_df['last_observed_cycle']].copy()
    test_with_rul = test_last_rows.merge(rul_df, on='unit')
    test_with_rul.drop('last_observed_cycle', axis=1, inplace=True)
    
    test_df.drop('last_observed_cycle', axis=1, inplace=True)
    
    return test_df, test_with_rul