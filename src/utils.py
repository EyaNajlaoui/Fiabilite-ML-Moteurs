import json
import numpy as np
import pandas as pd

def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_metrics(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def print_model_summary(model, model_name):
    print(f"\n=== {model_name.upper()} ===")
    if hasattr(model, 'summary'):
        model.summary()
    else:
        print(f"Type: {type(model).__name__}")
        if hasattr(model, 'get_params'):
            print("Parametres principaux:")
            params = model.get_params()
            for key in list(params.keys())[:5]:
                print(f"  {key}: {params[key]}")

def check_data_shapes(**kwargs):
    print("\n=== FORMES DES DONNEES ===")
    for name, data in kwargs.items():
        if hasattr(data, 'shape'):
            print(f"{name}: {data.shape}")
        else:
            print(f"{name}: {type(data)}")