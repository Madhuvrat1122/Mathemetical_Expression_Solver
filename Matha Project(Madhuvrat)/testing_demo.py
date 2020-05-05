import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.common.set_image_dim_ordering('th')
from keras.models import model_from_json
import pickle

json_file = open('john.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("john_final.h5")

pickle_in = open("dict1.pickle","rb")
example_dict1 = pickle.load(pickle_in)
train_data=example_dict1('test_images/new11.png')



pickle_in = open("dict.pickle","rb")
example_dict = pickle.load(pickle_in)
s=example_dict(train_data)
        
check=['A','B','C','D','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
old_s=s
for i in s:
    if i in check:
        s=s+'='+'0'
#for increement operator
if '++' in s:
    s=s.replace('++','+1')
#for decreement operator
elif '--' in s:
    s=s.replace('--','-1')
#for calculation    
print("{}={}".format(old_s,eval(s)))

    
