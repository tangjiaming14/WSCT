import csv
import pandas as pd
import myTools as mt
import numpy
csv_path = './data/fer2013/test.csv'
wlbl_path = './data/fer2013/wlbl_clu.csv'

psnr = 0.3

test_reader = pd.read_csv(csv_path,usecols=['emotion'])     #读取emotion的数据
#print(test_reader.head())
#print(test_reader.index)
#a = test_reader.loc[00000]
#b = int(a['emotion'])
#print(b)
test_values = test_reader.values
print(test_values)

wlbl = mt.perturb_labels(test_values, psnr)    #获取弱注释标签

print(wlbl)
#存储wlbl

save = pd.DataFrame(list(wlbl))
try:
    save.to_csv(wlbl_path, index = False)
except UnicodeEncodeError:
    print("编码错误, 该数据无法写到文件中, 直接忽略该数据")










