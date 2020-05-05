
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import pandas as pd

def load_images_from_folder(folder):
    train_data=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        img=~img
        if img is not None:
            ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            ctrs,ret=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
            w=int(28)
            h=int(28)
            maxi=0
            for c in cnt:
                x,y,w,h=cv2.boundingRect(c)
                maxi=max(w*h,maxi)
                if maxi==w*h:
                    x_max=x
                    y_max=y
                    w_max=w
                    h_max=h
            im_crop= thresh[y_max:y_max+h_max+10, x_max:x_max+w_max+10]
            im_resize = cv2.resize(im_crop,(28,28))
            im_resize=np.reshape(im_resize,(784,1))
            train_data.append(im_resize)
    return train_data


data=[]

data=load_images_from_folder('train images/-')
len(data)
for i in range(0,len(data)):
    data[i]=np.append(data[i],['10'])
    
print(len(data))




data0=load_images_from_folder('train images/0')
for i in range(0,len(data0)):
    data0[i]=np.append(data0[i],['0'])
data=np.concatenate((data,data0))
print(len(data))

data1=load_images_from_folder('train images/1')

for i in range(0,len(data1)):
    data1[i]=np.append(data1[i],['1'])
data=np.concatenate((data,data1))
print(len(data))

data100=load_images_from_folder('train images/+')

for i in range(0,len(data100)):
    data100[i]=np.append(data100[i],['11'])
data=np.concatenate((data,data100))
print(len(data))


data2=load_images_from_folder('train images/2')

for i in range(0,len(data2)):
    data2[i]=np.append(data2[i],['2'])
data=np.concatenate((data,data2))
print(len(data))

data3=load_images_from_folder('train images/3')

for i in range(0,len(data3)):
    data3[i]=np.append(data3[i],['3'])
data=np.concatenate((data,data3))
print(len(data))

data4=load_images_from_folder('train images/4')

for i in range(0,len(data4)):
    data4[i]=np.append(data4[i],['4'])
data=np.concatenate((data,data4))
print(len(data))

data5=load_images_from_folder('train images/5')

for i in range(0,len(data5)):
    data5[i]=np.append(data5[i],['5'])
data=np.concatenate((data,data5))
print(len(data))

data6=load_images_from_folder('train images/6')

for i in range(0,len(data6)):
    data6[i]=np.append(data6[i],['6'])
data=np.concatenate((data,data6))
print(len(data))

data7=load_images_from_folder('train images/7')

for i in range(0,len(data7)):
    data7[i]=np.append(data7[i],['7'])
data=np.concatenate((data,data7))
print(len(data))


data8=load_images_from_folder('train images/8')

for i in range(0,len(data8)):
    data8[i]=np.append(data8[i],['8'])
data=np.concatenate((data,data8))
print(len(data))

data9=load_images_from_folder('train images/9')

for i in range(0,len(data9)):
    data9[i]=np.append(data9[i],['9'])
data=np.concatenate((data,data9))
print(len(data))

data10=load_images_from_folder('train images/times')

for i in range(0,len(data10)):
    data10[i]=np.append(data10[i],['12'])
data=np.concatenate((data,data10))
print(len(data))

data11=load_images_from_folder('train images/A')

for i in range(0,len(data11)):
    data11[i]=np.append(data11[i],['13'])
data=np.concatenate((data,data11))
print(len(data))

data12=load_images_from_folder('train images/B')

for i in range(0,len(data12)):
    data12[i]=np.append(data12[i],['14'])
data=np.concatenate((data,data12))
print(len(data))

data13=load_images_from_folder('train images/C')

for i in range(0,len(data13)):
    data13[i]=np.append(data13[i],['15'])
data=np.concatenate((data,data13))
print(len(data))

data14=load_images_from_folder('train images/D')

for i in range(0,len(data14)):
    data14[i]=np.append(data14[i],['16'])
data=np.concatenate((data,data14))
print(len(data))

data15=load_images_from_folder('train images/E')

for i in range(0,len(data15)):
    data15[i]=np.append(data15[i],['17'])
data=np.concatenate((data,data15))
print(len(data))

data16=load_images_from_folder('train images/F')

for i in range(0,len(data16)):
    data16[i]=np.append(data16[i],['18'])
data=np.concatenate((data,data16))
print(len(data))

data17=load_images_from_folder('train images/G')

for i in range(0,len(data17)):
    data17[i]=np.append(data17[i],['19'])
data=np.concatenate((data,data17))
print(len(data))

data18=load_images_from_folder('train images/H')

for i in range(0,len(data18)):
    data18[i]=np.append(data18[i],['20'])
data=np.concatenate((data,data18))
print(len(data))

data19=load_images_from_folder('train images/I')

for i in range(0,len(data19)):
    data19[i]=np.append(data19[i],['21'])
data=np.concatenate((data,data19))
print(len(data))

data20=load_images_from_folder('train images/J')
for i in range(0,len(data20)):
    data20[i]=np.append(data20[i],['22'])
data=np.concatenate((data,data20))
print(len(data))

data21=load_images_from_folder('train images/K')
for i in range(0,len(data21)):
    data21[i]=np.append(data21[i],['23'])
data=np.concatenate((data,data21))
print(len(data))

data22=load_images_from_folder('train images/L')
for i in range(0,len(data22)):
    data22[i]=np.append(data22[i],['24'])
data=np.concatenate((data,data22))
print(len(data))

data23=load_images_from_folder('train images/M')
for i in range(0,len(data23)):
    data23[i]=np.append(data23[i],['25'])
data=np.concatenate((data,data23))
print(len(data))

data24=load_images_from_folder('train images/N')
for i in range(0,len(data24)):
    data24[i]=np.append(data24[i],['26'])
data=np.concatenate((data,data24))
print(len(data))

data25=load_images_from_folder('train images/O')
for i in range(0,len(data25)):
    data25[i]=np.append(data25[i],['27'])
data=np.concatenate((data,data25))
print(len(data))

data26=load_images_from_folder('train images/P')
for i in range(0,len(data26)):
    data26[i]=np.append(data26[i],['28'])
data=np.concatenate((data,data26))
print(len(data))

data27=load_images_from_folder('train images/Q')
for i in range(0,len(data27)):
    data27[i]=np.append(data27[i],['29'])
data=np.concatenate((data,data27))
print(len(data))

data28=load_images_from_folder('train images/R')
for i in range(0,len(data28)):
    data28[i]=np.append(data28[i],['30'])
data=np.concatenate((data,data28))
print(len(data))

data29=load_images_from_folder('train images/S')
for i in range(0,len(data29)):
    data29[i]=np.append(data29[i],['31'])
data=np.concatenate((data,data29))
print(len(data))

data30=load_images_from_folder('train images/T')
for i in range(0,len(data30)):
    data30[i]=np.append(data30[i],['32'])
data=np.concatenate((data,data30))
print(len(data))

data31=load_images_from_folder('train images/U')
for i in range(0,len(data31)):
    data31[i]=np.append(data31[i],['33'])
data=np.concatenate((data,data31))
print(len(data))

data32=load_images_from_folder('train images/V')
for i in range(0,len(data32)):
    data32[i]=np.append(data32[i],['34'])
data=np.concatenate((data,data32))
print(len(data))

data33=load_images_from_folder('train images/W')
for i in range(0,len(data33)):
    data33[i]=np.append(data33[i],['35'])
data=np.concatenate((data,data33))
print(len(data))

data34=load_images_from_folder('train images/X')
for i in range(0,len(data34)):
    data34[i]=np.append(data34[i],['36'])
data=np.concatenate((data,data34))
print(len(data))

data35=load_images_from_folder('train images/Y')
for i in range(0,len(data35)):
    data35[i]=np.append(data35[i],['37'])
data=np.concatenate((data,data35))
print(len(data))

data36=load_images_from_folder('train images/Z')
for i in range(0,len(data36)):
    data36[i]=np.append(data36[i],['38'])
data=np.concatenate((data,data36))
print(len(data))


df=pd.DataFrame(data,index=None)
df.to_csv('new_train_final.csv',index=False)