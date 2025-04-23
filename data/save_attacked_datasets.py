# scripts/save_attacked_ausgrid.py
"""
Loads the raw Ausgrid dataset from 'data/raw/ausgrid/', applies various synthetic
attacks month-by-month using the attack models from src/, and saves the original
and attacked dataframes to 'data/processed/ausgrid_attacked.h5'.
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_ausgrid_data, load_sgcc_data
from src.attack_models import get_attack_model, list_available_attacks

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_FILENAME_AUSGRID= 'ausgrid_attacked.h5'
OUTPUT_FILENAME_AUSGRID = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILENAME_AUSGRID)

OUTPUT_FILENAME_SGCC = 'sgcc_attacked.h5'
OUTPUT_FILENAME_SGCC = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILENAME_SGCC)

OUTPUT_FILENAME_SGCC_LABELS = 'original_labels.h5'
OUTPUT_FILENAME_SGCC_LABELS = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILENAME_SGCC_LABELS)


def main():
    """Loads data, applies attacks, and saves results."""
    ausgrid = load_ausgrid_data()
    sgcc, labels = load_sgcc_data()
    ausgrid.to_hdf(
        OUTPUT_FILENAME_AUSGRID,
        key='original',
        mode='w', 
    )
    sgcc.to_hdf(
        OUTPUT_FILENAME_SGCC,
        key='original',
        mode='w' 
    )

    labels.to_hdf(
        OUTPUT_FILENAME_SGCC_LABELS,
        key='original',
        mode='w'
    )





    for attack_id in list_available_attacks():        
        attack_model = get_attack_model(attack_id)
        attacked_ausgrid = attack_model.apply(ausgrid.copy(deep=True)) 
        attacked_sgcc = attack_model.apply(sgcc.copy(deep=True)) 
        save_key = f'attack_{attack_id}'
        
        attacked_ausgrid.to_hdf(
            OUTPUT_FILENAME_AUSGRID,
            key=save_key,
            mode='r+')
        
        attacked_sgcc.to_hdf(
            OUTPUT_FILENAME_SGCC,
            key=save_key,
            mode='r+')
        



if __name__ == "__main__":
    main()
