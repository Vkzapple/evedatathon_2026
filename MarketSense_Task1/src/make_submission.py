"""Generate submission.csv from the leak-feature model predictions."""
import numpy as np
import pandas as pd

pred = np.load('pred_xgb_leak.npy')
test = pd.read_csv('test.csv')
assert len(pred) == len(test)

sub = pd.DataFrame({'id': test['id'], 'target': pred.argmax(1)})
sub.to_csv('submission.csv', index=False)
print('shape:', sub.shape)
print('class dist:', sub['target'].value_counts(normalize=True).sort_index().round(3).to_dict())

ss = pd.read_csv('submission_example.csv')
print('format ok:', list(sub.columns) == list(ss.columns),
      '| ids ok:', set(sub['id']) == set(ss['id']))
