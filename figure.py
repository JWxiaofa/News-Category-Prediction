import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
df['category'].value_counts()
df['category'].value_counts().plot.bar(figsize=(20, 9))
plt.show()


