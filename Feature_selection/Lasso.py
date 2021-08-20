import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
from dimensional_reduction import lassodimension


data_train=pd.read_csv('DNA_sapce.csv')
data_=np.array(data_train)
data=data_[:,2:]
label=data_[:,1]
shu=scale(data)
data_2,index=lassodimension(shu,label)
shu=data_2
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('lasso.csv')
