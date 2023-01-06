import os
import pandas as pd
from typing import Literal

def make_dir(d):
  if not os.path.exists(d):
    os.makedirs(d)
  return d

def load_file(i: Literal[1, 2, 4]) -> pd.DataFrame:
  return pd.read_csv('Test%d.csv' % i)
