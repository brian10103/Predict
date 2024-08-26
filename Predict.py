# -*- coding: utf-8 -*-
"""
Created on Tue May 10 00:24:38 2022

@author: Brian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from imblearn.over_sampling import SMOTE
import seaborn as sns
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek 
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.utils.validation import check_is_fitted
from numba import cuda

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
 
dataset = pd.read_csv('D:/test/三年段/2016-2018DNN特徵_預測.csv')
#dataset = pd.read_csv('D:/test/三年段/2016-2018DNN特徵_分類.csv')
#dataset = pd.read_csv('D:/test/三年段/2019-2021DNN特徵_分類.csv')

    
X = dataset.drop(dataset.columns[[0]], axis = 1)
#X = X.drop(["Cooperation","Edges","A_Betweenesscentrality","A_Closnesscentrality","A_Degree","B_Betweenesscentrality","B_Closnesscentrality","B_Degree"],axis=1).values
X = X.drop(["Cooperation","Edges"],axis=1).values
y = dataset["Cooperation"].values   
    
    
#oversample = SMOTE(sampling_strategy= 0.3, random_state=42, k_neighbors = 5)
#oversample = BorderlineSMOTE(sampling_strategy= 0.5, random_state=42,kind="borderline-2")
oversample = SMOTEENN(sampling_strategy= 1, random_state=42)
#oversample = SMOTETomek(random_state=42)
X_resample, y_resample = oversample.fit_resample(X,y)
#X_resample, y_resample = oversample.fit_resample(X_train,y_train)

#ada = ADASYN(random_state=42)
#X_resample, y_resample = ada.fit_resample(X, y)    
    
y_resample=pd.DataFrame(y_resample)
X_resample=pd.DataFrame(X_resample)   
    
X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size = 0.3, random_state=42)

X_train = np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)   

X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.LongTensor(y_train).to(device)
y_test = torch.LongTensor(y_test).to(device)    
   
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))    
    
class Model(nn.Module):
    def __init__(self, input_features=14, hidden_layer1=24, hidden_layer2=21, output_features=2):
        super().__init__()
        self.fc1 = nn.Linear(input_features,hidden_layer1)                  
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)                  
        self.out = nn.Linear(hidden_layer2, output_features)      
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

model = Model()

# Make sure to call input = input.to(device) on any input tensors that you feed to the model
model.to(device)
model


from sklearn.model_selection import StratifiedKFold
import random
import time
from tqdm import tqdm
import os
#import tensorflow as tf
#from module import *
#import module

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #tf.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

kaeru_seed = 1337
seed_everything(seed=kaeru_seed)

batch_size = 1000
train_epochs = 1


splits = list(StratifiedKFold(n_splits=3, shuffle=True).split(X_train, y_train))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


train_preds = np.zeros((len(X_train)))
test_preds = np.zeros((len(X_test)))

seed_everything(kaeru_seed)

x_test_cuda = torch.tensor(X_test, dtype=torch.float32)
test = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)



for i, (train_idx, valid_idx) in enumerate(splits):
    x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.float32)
    y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32)
    x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.float32)
    y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32)
    
    model = model
    model
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0075)
    
    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    
    print(f'Fold {i + 1}')
    
    for epoch in range(train_epochs):
        start_time = time.time()
        
        model.train()
        avg_loss = 0.
        for x_batch, y_batch in tqdm(train_loader, disable=True):
            y_pred = model.forward(x_batch)
            y_batch = y_batch.type(torch.LongTensor)  # casting to long
            y_batch = y_batch.squeeze(1)    
            loss = loss_fn(y_pred.squeeze(1), y_batch.squeeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        
        model.eval()
        valid_preds_fold = np.zeros((x_val_fold.size(0)))
        test_preds_fold = np.zeros(len(X_test))
        avg_val_loss = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            y_pred = model.forward(x_batch)
            y_batch = y_batch.type(torch.LongTensor)  # casting to long
            y_batch = y_batch.squeeze(1)  
            avg_val_loss += loss_fn(y_pred.squeeze(1), y_batch.squeeze(1)).item() / len(valid_loader)
            valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().detach().numpy())[:, 0]
        
        elapsed_time = time.time() - start_time 
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
            epoch + 1, train_epochs, avg_loss, avg_val_loss, elapsed_time))
        
    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    train_preds[valid_idx] = valid_preds_fold
    test_preds += test_preds_fold / len(splits)




from sklearn.metrics import accuracy_score

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = accuracy_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'accuracy_score': best_score}
    return search_result



search_result = threshold_search(y_train, train_preds)
print(search_result)



preds = []
valid_loss = 0
accuracy = 0
with torch.no_grad():
    for val in X_test:
        y_hat = model.forward(val)
        preds.append(y_hat.argmax().item())




df = pd.DataFrame({'Y': list(y_test.cpu().numpy()), 'pred': preds})
df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['pred'])]
df



df['Correct'].sum() / len(df)
print(f"accuracy: {df['Correct'].sum() / len(df)}")


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(list(y_test.cpu().numpy()), preds, labels=None, sample_weight=None)
tn, fp, fn, tp = confusion_matrix(list(y_test.cpu().numpy()), preds).ravel()


Accuracy = (tp+tn)/(tp+fp+fn+tn)
Precision = tp/(tp+fp)
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
F1 = 2 * Precision * sensitivity/(Precision+sensitivity)


print("Accuracy:" + str(Accuracy))
print("Precision:" + str(Precision))
print("Recall:" + str(sensitivity))
print("F1:" + str(F1))

"""
#計算roc_curve, auc
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder

plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')

# One hot encoding
enc = OneHotEncoder()
#Y_onehot = enc.fit_transform(y_test[:, np.newaxis]).toarray()

with torch.no_grad():
    y_pred = model(X_test.cpu()).numpy()
    fpr, tpr, threshold = roc_curve(list(y_test.cpu().numpy()),preds)
    #fpr, tpr, threshold = roc_curve(list(y_test.cpu().numpy()), preds)
    #list(y_test.cpu().numpy()), preds
    #Y_onehot.ravel(), y_pred.ravel()
plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc(fpr, tpr)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
#計算roc_curve, auc
"""


#計算特徵重要性(SHAP法)
import shap

#attrib_data = X_train[:526]
attrib_data = X_train[np.random.choice(X_train.shape[1], 10000)]
#attrib_data = X_train
explainer = shap.DeepExplainer(model, attrib_data)
num_explanations = 1000
explain_data = X_test[np.random.choice(X_test.shape[1], num_explanations)]
shap_vals = explainer.shap_values(explain_data)
#shap_vals = explainer.shap_values(X_test)

shap.summary_plot(shap_vals[1], feature_names=['A_Firms_Patents','A_Firms_Inventors','A_Firms_Inventor_Country','A_Firms_Cited_by_Patent',
                  'A_Firms_Granted','B_Firms_Patents','B_Firms_Inventors','B_Firms_Inventor_Country',
                  'B_Firms_Cited_by_Patent','B_Firms_Granted','Firm_Country','A_Technological_Breadth',
                  'B_Technological_Breadth','Technological_Similarity'],  plot_type="bar")

"""
#改
shap.summary_plot(shap_vals, feature_names=['B_T','A_P','B_C','A_T',
                  'B_F','A_C','S','B_I',
                  'C','A_F','B_P','A_I',
                  'B_G','A_G'], plot_type="bar")
shap.summary_plot(shap_vals[1], feature_names=['B_Technological_Breadth','A_Firms_Cited_by_Patent','B_Firms_Inventor_Country','A_Technological_Breadth',
                  'B_Firms_Granted','A_Firms_Inventor_Country','Firm_Country','B_Firms_Inventors',
                  'A_Firms_Inventors','A_Firms_Patents','B_Firms_Cited_by_Patent','Technological_Similarity',
                  'B_Firms_Patents','A_Firms_Granted'], plot_type="bar")

#原
shap.summary_plot(shap_vals[1], feature_names=['A_F','A_I','A_C','A_P',
                  'A_G','B_F','B_I','B_C',
                  'B_P','B_G','C','A_T',
                  'B_T','S'], plot_type="bar")

shap.summary_plot(shap_vals, feature_names=['A_Firms_Patents','A_Firms_Inventors','A_Firms_Inventor_Country','A_Firms_Cited_by_Patent',
                  'A_Firms_Granted','B_Firms_Patents','B_Firms_Inventors','B_Firms_Inventor_Country',
                  'B_Firms_Cited_by_Patent','B_Firms_Granted','Firm_Country','A_Technological_Breadth',
                  'B_Technological_Breadth','Technological_Similarity'],  plot_type="bar")
"""

print("END")

"""
dataset_Pr = pd.read_csv('D:/test/三年段/2019-2021DNN特徵_預測.csv')
#dataset_Pr = pd.read_csv('D:/test/三年段/2019-2021DNN特徵_分類.csv')

Pr = dataset_Pr.drop(dataset_Pr.columns[[0]], axis = 1)
#Pr = Pr.drop(["Cooperation","Edges","A_Betweenesscentrality","A_Closnesscentrality","A_Degree","B_Betweenesscentrality","B_Closnesscentrality","B_Degree"],axis=1).values
Pr = Pr.drop(["Cooperation","Edges"],axis=1).values

Pr=np.array(Pr)
Pr = torch.FloatTensor(Pr).to(device)

predicts = []
valid_loss = 0
accuracy = 0
with torch.no_grad():
    for val in Pr:
        y_pre = model.forward(val)
        predicts.append(y_pre.argmax().item())
        
predicts_df = pd.DataFrame({'Edges': dataset_Pr["Edges"], 'predicts': predicts})


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(dataset_Pr["Cooperation"], predicts, labels=None, sample_weight=None)
tn, fp, fn, tp = confusion_matrix(dataset_Pr["Cooperation"], predicts).ravel()


Accuracy = (tp+tn)/(tp+fp+fn+tn)
Precision = tp/(tp+fp)
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
F1 = 2 * Precision * sensitivity/(Precision+sensitivity)


print("Accuracy:" + str(Accuracy))
print("Precision:" + str(Precision))
print("Recall:" + str(sensitivity))
print("F1:" + str(F1))


predicts_df.to_csv('D:/test/三年段/0524/預測/ENN1921_0627預測.csv')       
     
"""


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

"""
### PLOT CORRELATION MATRIX ###
DD = dataset.drop(dataset.columns[[0]], axis = 1)


plt.figure(figsize=(30,30))
train_size = int(DD.shape[0]*0.8)
corr_matrix = DD.iloc[:train_size,:].corr().abs()
sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot=True)   
        
  

### FIT GRADIENTBOOSTING ###

gb = GradientBoostingRegressor(n_estimators=100)
gb.fit(X_train.cpu().numpy(), y_train.cpu().numpy())


### PREDICTION ERROR ON TEST DATA ###


### FEATURE IMPORTANCES REPORT ###
plt.figure(figsize=(15, 8))  # 設定新圖表大小

df = pd.DataFrame(dict(
    names=['A_Firms_Patents','A_Firms_Inventors','A_Firms_Inventor_Country','A_Firms_Cited_by_Patent',
                  'A_Firms_Granted','B_Firms_Patents','B_Firms_Inventors','B_Firms_Inventor_Country',
                  'B_Firms_Cited_by_Patent','B_Firms_Granted','Firm_Country','A_Technological_Breadth',
                  'B_Technological_Breadth','Technological_Similarity'],
      feature=gb.feature_importances_))

df_sorted = df.sort_values('feature')
plt.barh('names', 'feature', data=df_sorted)    
              
#plt.barh(range(X_train.shape[1]), gb.feature_importances_)

plt.yticks(range(X_train.shape[1]), ['A_Firms_Patents','A_Firms_Inventors','A_Firms_Inventor_Country','A_Firms_Cited_by_Patent',
                  'A_Firms_Granted','B_Firms_Patents','B_Firms_Inventors','B_Firms_Inventor_Country',
                  'B_Firms_Cited_by_Patent','B_Firms_Granted','Firm_Country','A_Technological_Breadth',
                  'B_Technological_Breadth','Technological_Similarity'])
"""
"""
plt.bar(range(X_train.shape[1]), gb.feature_importances_)

plt.xticks(range(X_train.shape[1]), ['A_F','A_I','A_C','A_P',
                  'A_G','B_F','B_I','B_C',
                  'B_P','B_G','C','A_T',
                  'B_T','S'])
"""
"""
plt.xticks(range(X_train.shape[1]), ['A_Firms_Nol_Firms','A_Firms_Nol_Inventors','A_Firms_Nol_Inventor_Country','A_Firms_Nol_Cited_by_Patent_Count',
                  'A_Firms_Nol_Num_Granted','B_Firms_Nol_Firms','B_Firms_Nol_Inventors','B_Firms_Nol_Inventor_Country',
                  'B_Firms_Nol_Cited_by_Patent_Count','B_Firms_Nol_Num_Granted','Firms_Nol_Firm_Country','A_Technological_Breadth',
                  'B_Technological_Breadth','Technological_Similarity'])

plt.xticks(range(X_train.shape[1]), ['A_專利數','A_發明人數','A_發明人國籍','A_被引用數',
                  'A_授權數','B_專利數','B_發明人數','B_發明人國籍',
                  'B_被引用數','B_授權數','組織國籍','A_廣度',
                  'B_廣度','相似度'])

np.set_printoptions(False)      


from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=100)
xgb.fit(X_train, y_train)

xgb.feature_importances_

dx = pd.DataFrame(dict(
    names=['A_Firms_Patents','A_Firms_Inventors','A_Firms_Inventor_Country','A_Firms_Cited_by_Patent',
                  'A_Firms_Granted','B_Firms_Patents','B_Firms_Inventors','B_Firms_Inventor_Country',
                  'B_Firms_Cited_by_Patent','B_Firms_Granted','Firm_Country','A_Technological_Breadth',
                  'B_Technological_Breadth','Technological_Similarity'],
      feature=xgb.feature_importances_))

dx_sorted = dx.sort_values('feature')
plt.barh('names', 'feature', data=dx_sorted)    

import shap
#explainer = shap.TreeExplainer(xgb)
#shap_values = explainer.shap_values(X_test[cols])
"""