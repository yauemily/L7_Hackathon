import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_root = "https://github.com/yauemily/L7_Hackathon/raw/main/Data/"
data = pd.read_csv(data_root + "corona_tested_individuals_ver_006.english.csv")
data.head(5)