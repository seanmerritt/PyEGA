from EGA import EGA
import pandas as pd

ega = EGA()

df = pd.read_csv(r'C:\Users\seanm\Desktop\PyEGA\Data\intelligencebattery.csv')
#df = pd.DataFrame([[1,2,3,4,5],[2,4,6,8,10],[6,7,5,8,4],[5,4,3,2,1]])
#print(df)

ega.fit(df.iloc[:,8:])