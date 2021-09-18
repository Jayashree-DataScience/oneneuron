import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import joblib
from matplotlib.colors import ListedColormap
plt.style.use("fivethirtyeight")

def prepare_data(df):
    x=df.drop("y",axis=1)
    y=df["y"]
    return x,y

def save_model(model,filename):
    model_dir="models"
    os.makedirs(model_dir,exist_ok=Ture)
    filePath=os.path.join(model_dir,filename)
    joblib.dump(model,filePath)