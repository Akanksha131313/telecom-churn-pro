import importlib.util
import sys
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import plotly.express as px

# ---------------- Import your src modules dynamically ----------------
src_path = pathlib.Path(__file__).parent / "03_src"

# Module 01_data_preprocessing
spec = importlib.util.spec_from_file_location("data_preprocessing", src_path / "01_data_preprocessing.py")
dp = importlib.util.module_from_spec(spec)
sys.modules["data_preprocessing"] = dp
spec.loader.exec_module(dp)

# Module 02_model_evaluation
spec2 = importlib.util.spec_from_file_location("model_evaluation", src_path / "02_model_evaluation.py")
me = importlib.util.module_from_spec(spec2)
sys.modules["model_evaluation"] = me
spec2.loader.exec_module(me)

# Module 03_model_training
spec3 = importlib.util.spec_from_file_location("model_training", src_path / "03_model_training.py")
mt = importlib.util.module_from_spec(spec3)
sys.modules["model_training"] = mt
spec3.loader.exec_module(mt)
