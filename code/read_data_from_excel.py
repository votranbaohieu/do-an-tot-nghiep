import pandas as pd

f = r'./data/data-training.xlsx'


data = pd.read_excel(f)
df = pd.DataFrame(data)
print(df)
