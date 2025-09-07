import os
import pandas as pd

def test_tdcs_has_three_class_columns():
    root = 'train/tdcsfog'
    samples = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.csv')][:1]
    assert samples, 'no tdcs csvs found'
    df = pd.read_csv(samples[0], nrows=1)
    for c in ['StartHesitation', 'Turn', 'Walking']:
        assert c in df.columns, f'missing {c} in tdcs csv'
