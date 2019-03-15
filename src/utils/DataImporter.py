import pandas as pd
import matplotlib.pyplot as plt

def import_data(filename):
    df = pd.read_csv('./resources/' + filename,delimiter=';')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop(['Open','High','Low','Adj Close','Volume'], axis=1)
    return df
