from keras.layers import Dense, Dropout 
from keras.layers.recurrent import LSTM, GRU 
from keras.models import Sequential 
import pandas as pd 
import numpy as np 
from keras.layers import Flatten 
from keras.layers import GRU,Bidirectional
from keras.layers import Conv1D, MaxPooling1D
import matplotlib.pyplot as plt 
from sklearn.preprocessing import scale 
from sklearn.metrics import roc_curve, auc 
from sklearn.model_selection import StratifiedKFold 
import utils.tools as utils 
 
 
model = Sequential() 
model.add(Bidirectional(GRU(8,return_sequences=True))) 
model.add(Dropout(0.5)) 
model.add(Bidirectional(GRU(4,return_sequences=True)))
model.add(Dropout(0.5)) 
model.add(Flatten()) 


def get_CNN_model(input_dim,out_dim):
    model = Sequential()
    model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation= 'relu'))
    model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))
    model.add(Conv1D(filters = 32, kernel_size =  3, padding = 'same', activation= 'relu'))
    model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME")) 
    model.add(Flatten())
    model.add(Dense(int(input_dim/4), activation = 'relu'))
    model.add(Dense(int(input_dim/8), activation = 'relu'))
    model.add(Dense(out_dim, activation = 'sigmoid',name="Dense_2"))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])
    return model


model.add(Dense(2, activation = 'sigmoid',name="Dense_2")) 
model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics =['accuracy'])#rmsprop 




data_=pd.read_csv(r'DNA_vector.csv') 
data=np.array(data_) 
data=data[:,2:] 
[m1,n1]=np.shape(data) 
label1=np.ones((int(m1/2),1))
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2) 
shu=scale(data)
X1=shu 
y=label 
X=np.reshape(X1,(-1,1,n1))  
sepscores = []  
ytest=np.ones((1,2))*0.5 
yscore=np.ones((1,2))*0.5 

 
skf= StratifiedKFold(n_splits=10) 

 
for train, test in skf.split(X,y):  
    y_train=utils.to_categorical(y[train])
    cv_clf = model 
    hist=cv_clf.fit(X[train],  
                    y_train, 
                    epochs=100) 
     
    y_score=cv_clf.predict(X[test])
    y_class= utils.categorical_probas_to_classes(y_score) 
    
     
    y_test=utils.to_categorical(y[test]) 
    ytest=np.vstack((ytest,y_test)) 
    y_test_tmp=y[test]        
    yscore=np.vstack((yscore,y_score)) 
     
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp) 
    fpr, tpr, _ = roc_curve(y_test[:,1], y_score[:,1]) 
    roc_auc = auc(fpr, tpr) 
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc]) 
     
scores=np.array(sepscores) 
result1=np.mean(scores,axis=0) 
H1=result1.tolist() 
sepscores.append(H1) 
result=sepscores 

 
row=yscore.shape[0] 
yscore=yscore[np.array(range(1,row)),:] 
yscore_sum = pd.DataFrame(data=yscore) 
yscore_sum.to_csv('yscore.csv') 

 
ytest=ytest[np.array(range(1,row)),:] 
ytest_sum = pd.DataFrame(data=ytest) 
ytest_sum.to_csv('ytest.csv') 

 

 
data_csv = pd.DataFrame(data=result) 
data_csv.to_csv('BiGRU_CNN.csv') 
