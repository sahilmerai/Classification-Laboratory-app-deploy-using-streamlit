
# import library require

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.evaluate import bias_variance_decomp
from sklearn.preprocessing import binarize
from sklearn.metrics import accuracy_score , confusion_matrix ,classification_report ,roc_auc_score,auc,roc_curve
from sklearn.metrics import recall_score,precision_score,f1_score,fbeta_score,plot_roc_curve
from mlxtend.evaluate import mcnemar
from sklearn.model_selection import RandomizedSearchCV
import plotly.graph_objects as go

# title 
st.title('Classification Laboratory')

# import csv file
dataset = st.sidebar.file_uploader("upload file here", type = ['csv'])
if dataset is not None:
    df = pd.read_csv(dataset) 
    pr = ProfileReport(df)  
    col = df.columns.values
st.sidebar.write("try with this dataset")
st.sidebar.write("https://github.com/sahilmerai/ml_basic/blob/3ffaaae1847a86f3bd526a5d070155949094bda7/dd.csv") 

# EDA part 
Data_view = st.sidebar.checkbox('show data')

if Data_view:
    st.write('## Data set')
    st.dataframe(df.head(5))
    st.write(df.shape)

EDA = st.sidebar.checkbox('EDA')

if EDA:
    st.write('EDA')
    st_profile_report(pr)


# select varible 
st.sidebar.subheader("select target variable")
target_variable = st.sidebar.selectbox('select target variable',(col))
# define variable
X = df.drop(target_variable,axis=1)
y = df[target_variable]

# train test split part
st.sidebar.subheader("Train test split")
size = st.sidebar.slider('Test size', 0.0, 1.00, 0.3)
random = st.sidebar.number_input('Random seed',min_value=5)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=size, random_state=random)
shapee = st.sidebar.checkbox("Shape of train and test data",)
if shapee:
     st.write('shape of x train data',X_train.shape)
     st.write('shape of x test data',X_test.shape)
     st.write('shape of y test data',y_test.shape)
     st.write("shape of y train data",y_train.shape)

# define scaled variable
st.sidebar.subheader("Feature Engineering")
st.sidebar.caption("PCA")
X_scaled = df.drop(target_variable,axis=1)
y_scaled = df[target_variable]
X_trainS, X_testS, y_trainS, y_testS = train_test_split( X_scaled, y_scaled, test_size=size, random_state=random)

scaler = StandardScaler()
scaler.fit(X_trainS)
scaler.fit(X_testS)
X_trainS = scaler.transform(X_trainS)
X_testS = scaler.transform(X_testS)
scaler.fit(X_scaled)
scaler_data = scaler.transform(X_scaled)

# PCA

pca = PCA(n_components=3)
pca.fit(scaler_data)

x_pca = pca.transform(scaler_data)

scores_df = pd.DataFrame(x_pca, columns=['PC1', 'PC2', 'PC3'])

Y_label = []

for i in y_scaled:
  if i == 0:
    Y_label.append('live')
  elif i == 1:
    Y_label.append('dead')
  else:
    Y_label.append('3rd value')

status_1 = pd.DataFrame(Y_label, columns=['status'])

df_scores = pd.concat([scores_df, status_1], axis=1)

loadings = pca.components_.T
df_loadings = pd.DataFrame(loadings, columns=['PC1', 'PC2','PC3'], index=X_scaled.columns)
explained_variance = pca.explained_variance_ratio_
#explained_variance
explained_variance = np.insert(explained_variance, 0, 0)
cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))

pc_df = pd.DataFrame(['','PC1', 'PC2', 'PC3'], columns=['PC'])
explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])
cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=['Cumulative Variance'])

df_explained_variance = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis=1)
#df_explained_variance

## plot
figg = px.bar(df_explained_variance, 
             x='PC', y='Explained Variance',
             text='Explained Variance',
             width=800)

figg.update_traces(texttemplate='%{text:.3f}', textposition='outside')
## plot

fig = px.scatter_3d(df_scores, x='PC1', y='PC2', z='PC3',
              color='status')

## plot
Graph = st.sidebar.checkbox('3d graph')
if Graph:
    fig.show()
    #st.pydeck_chart(fig)

Graph3 = st.sidebar.checkbox('Explained variance in bar plot')
if Graph3:
    figg.show()
    #st.bar_chart(figg)

Graph1 = st.sidebar.checkbox('Explained variance')
if Graph1:
    st.write(explained_variance)

st.sidebar.subheader("Model Manager")
KNN = st.sidebar.checkbox("K Nearest Neighbour") 
if KNN:
    n_neighbors = st.sidebar.slider('max_depth', 2, 25,5)
    KNN = KNeighborsClassifier(n_neighbors=n_neighbors)
    KNN.fit(X_trainS,y_trainS)
    Y_pre = KNN.predict(X_testS)
    # Accuracy = accuracy_score(y_test,Y_pre)
    # Precision = precision_score(y_test,Y_pre)
    # Recall  = recall_score(y_test,Y_pre)
    # F1_score = f1_score(y_test,Y_pre)
    # Fbeta  = fbeta_score(y_test,Y_pre,average='macro', beta=0.5)
    Counfusion_metrix = confusion_matrix(y_testS,Y_pre)
    Classification_report = classification_report(y_testS,Y_pre)
    clsreport = print(Classification_report)
    figgg, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(Counfusion_metrix), annot=True, cmap="YlGnBu" ,fmt='g',ax=ax)
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        KNN, X_train.values, y_train.values, X_test.values, y_test.values, 
        loss='0-1_loss',
        random_seed=random)
    
    bias = st.checkbox("Bias_variance_decomp")
    if bias:
        st.write('Average expected loss: %.3f' % avg_expected_loss)
        st.write('Average bias: %.3f' % avg_bias)
        st.write('Average variance: %.3f' % avg_var)
    report = st.checkbox("Classification_report")
    if report:
        st.text(Classification_report)
    metrix = st.checkbox(" Counfusion_metrix")
    if metrix:
        st.write(figgg)

    metrix_T1 = []
    for n in range(1,25):
        KNN = KNeighborsClassifier(n_neighbors=n)
        KNN.fit(X_trainS,y_trainS)
        Y_pre = KNN.predict(X_testS)
        Counfusion_metrix = confusion_matrix(y_testS,Y_pre)
        FN = Counfusion_metrix[1][0]
        metrix_T1.append(FN)
    kn = pd.DataFrame({'K':range(1,25),"type 2":metrix_T1})
    kn = kn[kn['type 2'] == kn['type 2'].min()]
    
    if st.checkbox('click here to get best parameter'):
     st.dataframe(kn.head())
    else:
     st.sidebar.write("")

    chi2, p = mcnemar(ary=Counfusion_metrix, exact=True)
    mcn = st.checkbox("Mc nemar test")
    if mcn:
         st.write('chi-squared:', chi2)
         st.write('p-value:', p)

    y_pred_proba_clf = KNN.predict_proba(X_testS)
    y_pred_proba_clf =  y_pred_proba_clf[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_clf)
    r_auc = roc_auc_score(y_test, y_pred_proba_clf)

    # Seaborn's beautiful styling
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
    fig2=plt.figure(figsize=(25, 13))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='DT (AUROC = %0.3f)' % r_auc,marker='*')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend()

    ROC = st.checkbox("ROC CURVE")
    if ROC:
        st.pyplot(fig2)
        st.write("AUC:",r_auc)

LR = st.sidebar.checkbox("Logistic Regression")
if LR:
    thr = st.sidebar.slider("thresold",0.00,1.00,0.5)
    LR = LogisticRegression(random_state=random)
    LR.fit(X_train,y_train)
    Y_pre = LR.predict(X_test)
    y_pred_proba_LR = LR.predict_proba(X_test)
    y_pred_default_LR = binarize(y_pred_proba_LR, threshold=thr)
    # Accuracy = accuracy_score(y_test,Y_pre)
    # Precision = precision_score(y_test,Y_pre)
    # Recall  = recall_score(y_test,Y_pre)
    # F1_score = f1_score(y_test,Y_pre)
    # Fbeta  = fbeta_score(y_test,Y_pre,average='macro', beta=0.5)
    Counfusion_metrix = confusion_matrix(y_test,y_pred_default_LR[:,1])
    Classification_report = classification_report(y_test,Y_pre)
    clsreport = print(Classification_report)
    figgg, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(Counfusion_metrix), annot=True, cmap="YlGnBu" ,fmt='g',ax=ax)
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        LR, X_train.values, y_train.values, X_test.values, y_test.values, 
        loss='0-1_loss',
        random_seed=random)
    
    bias = st.checkbox("Bias_variance_decomp")
    if bias:
        st.write('Average expected loss: %.3f' % avg_expected_loss)
        st.write('Average bias: %.3f' % avg_bias)
        st.write('Average variance: %.3f' % avg_var)

    report = st.checkbox("Classification_report")
    if report:
        st.text(Classification_report)
    metrix = st.checkbox(" Counfusion_metrix")
    if metrix:
        st.write(figgg)
    
    chi2, p = mcnemar(ary=Counfusion_metrix, exact=True)
    mcn = st.checkbox("Mc nemar test")
    if mcn:
         st.write('chi-squared:', chi2)
         st.write('p-value:', p)

    y_pred_proba_clf = LR.predict_proba(X_test)
    y_pred_proba_clf =  y_pred_proba_clf[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_clf)
    r_auc = roc_auc_score(y_test, y_pred_proba_clf)

    # Seaborn's beautiful styling
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
    fig2=plt.figure(figsize=(25, 13))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='DT (AUROC = %0.3f)' % r_auc,marker='*')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend()

    ROC = st.checkbox("ROC CURVE")
    if ROC:
        st.pyplot(fig2)
        st.write("AUC:",r_auc)
    
DT = st.sidebar.checkbox('DecisionTree')
if DT:
    criterion = st.sidebar.select_slider("chose criterion",options=["gini","entropy"])     
    max_depth = st.sidebar.slider('max_depth', 2, 25,2)
    max_leaf_nodes = st.sidebar.slider('max_leaf_nodes', 2, 25,2)
    DT = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,random_state=random)
    DT.fit(X_train,y_train)
    Y_pre = DT.predict(X_test)
    Counfusion_metrix = confusion_matrix(y_test,Y_pre)
    Classification_report = classification_report(y_test,Y_pre)
    clsreport = print(Classification_report)
    figgg, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(Counfusion_metrix), annot=True, cmap="YlGnBu" ,fmt='g',ax=ax)
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        DT, X_train.values, y_train.values, X_test.values, y_test.values, 
        loss='0-1_loss',
        random_seed=random)
    
    bias = st.checkbox("Bias_variance_decomp")
    if bias:
        st.write('Average expected loss: %.3f' % avg_expected_loss)
        st.write('Average bias: %.3f' % avg_bias)
        st.write('Average variance: %.3f' % avg_var)



    report = st.checkbox("Classification_report")
    if report:
        st.text(Classification_report)
    metrix = st.checkbox(" Counfusion_metrix")
    if metrix:
        st.write(figgg)
    
    ##### best parameter ###
    
    metrix_T1 = []
    a = []
    b = []
    n=25
    for one in range(2,n+1):
        for two in range(2,n+1):
        # print(one,two)
        # for nodes_depth in range(one,two):
            DT = DecisionTreeClassifier(  criterion=criterion , max_leaf_nodes=one,max_depth=two , random_state=random)
            DT.fit(X_train,y_train)
            Y_pre = DT.predict(X_test)
            fn = confusion_matrix(y_test,Y_pre)
            FN = fn[1][0]
            metrix_T1.append(FN)
            a.append(one)
            b.append(two)

    best  = pd.DataFrame({'max_leaf_nodes':a ,'max_depth':b, 'type 2 error':metrix_T1 })
    grid_contour1 = best.groupby(['max_leaf_nodes','max_depth']).mean()
    grid_reset = grid_contour1.reset_index()
    grid_reset.columns = ['max_leaf_nodes', 'max_depth', 'type 2 error']
    grid_pivot = grid_reset.pivot('max_leaf_nodes', 'max_depth')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    layout = go.Layout(
            xaxis=go.layout.XAxis(
              title=go.layout.xaxis.Title(
              text='max_leaf_nodes')
             ),
             yaxis=go.layout.YAxis(
              title=go.layout.yaxis.Title(
              text='max_depth') 
            ) )

    fig1 = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
    fig1.update_layout(title='best',
                  scene = dict(
                    xaxis_title='max_leaf_nodes',
                    yaxis_title='max_depth',
                    zaxis_title='type 2 error'),
                  autosize=False,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65, t=90))      

    Graph4 = st.checkbox('best perameter')
    if Graph4:
        fig1.show()

    y_pred_proba_clf = DT.predict_proba(X_test)
    y_pred_proba_clf =  y_pred_proba_clf[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_clf)
    r_auc = roc_auc_score(y_test, y_pred_proba_clf)


    # Seaborn's beautiful styling
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
    fig2=plt.figure(figsize=(25, 13))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='DT (AUROC = %0.3f)' % r_auc,marker='*')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend()
    # plt.show()

    ROC = st.checkbox("ROC CURVE")
    if ROC:
        st.pyplot(fig2)
        st.write("AUC:",r_auc)
    
    chi2, p = mcnemar(ary=fn, exact=True)
    mcn = st.checkbox("Mc nemar test")
    if mcn:
         st.write('chi-squared:', chi2)
         st.write('p-value:', p)
    

RF = st.sidebar.checkbox('RandomForestClassifier')
if RF:
    criterion = st.sidebar.select_slider("chose criterion",options=["gini","entropy"])
    boot = st.sidebar.select_slider("chose criterion",options=["True","False"])
    max_f = st.sidebar.select_slider("chose criterion",options=["auto","sqrt",'log2'])
    min_s_S = st.sidebar.slider('min_sample_split', 2, 10,5)
    min_s_l = st.sidebar.slider('min_sample_leaf', 1, 5,2)
    n_jobs = st.sidebar.slider('n_jobs', -1, 50,-1)
    max_depth = st.sidebar.slider('max_depth', 2, 50,5)
    n_estimators = st.sidebar.slider('n_estimators', 1, 500,100)

    RF = RandomForestClassifier(criterion=criterion,max_depth=max_depth,n_estimators=n_estimators,random_state=random,
                                    max_features=max_f,bootstrap=boot,min_samples_split=min_s_S,
                                    min_samples_leaf=min_s_l,n_jobs=n_jobs)
    RFF = RandomForestClassifier()
    RF.fit(X_train,y_train)
    Y_pre = RF.predict(X_test)
    Counfusion_metrix = confusion_matrix(y_test,Y_pre)
    Classification_report = classification_report(y_test,Y_pre)
    clsreport = print(Classification_report)
    figgg, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(Counfusion_metrix), annot=True, cmap="YlGnBu" ,fmt='g',ax=ax)
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
    RF, X_train.values, y_train.values, X_test.values, y_test.values, 
    loss='0-1_loss',
    random_seed=random)
    
    bias = st.checkbox("Bias_variance_decomp")
    if bias:
        st.write('Average expected loss: %.3f' % avg_expected_loss)
        st.write('Average bias: %.3f' % avg_bias)
        st.write('Average variance: %.3f' % avg_var)
    report = st.checkbox("Classification_report")
    if report:
        st.text(Classification_report)
    metrix = st.checkbox(" Counfusion_metrix")
    if metrix:
        st.write(figgg)

    # Number of trees in random forest
    criterion = ['gini','entropy']
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt','log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    n_jobs = [-1,25,50]
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'n_jobs':n_jobs,
               'criterion' : criterion,
               'bootstrap': bootstrap}
    rf_random = RandomizedSearchCV(estimator = RFF, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=random, n_jobs = -1)
    rf_random.fit(X_train, y_train)
    hyper = rf_random.best_params_

    if st.checkbox('click here to get best parameter'):
     st.json(hyper)
    else:
     st.sidebar.write("")

    y_pred_proba_clf = RF.predict_proba(X_test)
    y_pred_proba_clf =  y_pred_proba_clf[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_clf)
    r_auc = roc_auc_score(y_test, y_pred_proba_clf)


    # Seaborn's beautiful styling
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
    fig2=plt.figure(figsize=(25, 13))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='DT (AUROC = %0.3f)' % r_auc,marker='*')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend()
    # plt.show()

    ROC = st.checkbox("ROC CURVE")
    if ROC:
        st.pyplot(fig2)
        st.write("AUC:",r_auc)

    chi2, p = mcnemar(ary=Counfusion_metrix, exact=True)
    mcn = st.checkbox("Mc nemar test")
    if mcn:
         st.write('chi-squared:', chi2)
         st.write('p-value:', p)
        
ADA = st.sidebar.checkbox("AdaBoostClassifier")
if ADA:
    criterion = st.sidebar.select_slider("chose criterion",options=["gini","entropy"])     
    max_depth = st.sidebar.slider('max_depth', 2, 25,2)
    max_leaf_nodes = st.sidebar.slider('max_leaf_nodes', 2, 25,2)
    DT = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,random_state=random)
    DT.fit(X_train,y_train)
    Y_pre = DT.predict(X_test)
    ADA = AdaBoostClassifier(base_estimator=DT,random_state=random)
    ADA.fit(X_train,y_train)
    Y_pre = ADA.predict(X_test)
    Accuracy = accuracy_score(y_test,Y_pre)
    Precision = precision_score(y_test,Y_pre)
    Recall  = recall_score(y_test,Y_pre)
    F1_score = f1_score(y_test,Y_pre)
    Fbeta  = fbeta_score(y_test,Y_pre,average='macro', beta=0.5)
    Counfusion_metrix = confusion_matrix(y_test,Y_pre)
    Classification_report = classification_report(y_test,Y_pre)
    clsreport = print(Classification_report)
    figgg, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(Counfusion_metrix), annot=True, cmap="YlGnBu" ,fmt='g',ax=ax)
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        ADA, X_train.values, y_train.values, X_test.values, y_test.values, 
        loss='0-1_loss',
        random_seed=random)
    
    bias = st.checkbox("Bias_variance_decomp")
    if bias:
        st.write('Average expected loss: %.3f' % avg_expected_loss)
        st.write('Average bias: %.3f' % avg_bias)
        st.write('Average variance: %.3f' % avg_var)
    report = st.checkbox("Classification_report")
    if report:
        st.text(Classification_report)
    metrix = st.checkbox(" Counfusion_metrix")
    if metrix:
        st.write(figgg)

    metrix_T1 =[]
    a = []
    b = []
    n=25
    for one in range(2,n+1):
        for two in range(2,n+1):
            DT = DecisionTreeClassifier(criterion=criterion,max_depth=two,max_leaf_nodes=one,random_state=5)
            DT.fit(X_train,y_train)
            Y_pree = DT.predict(X_test)
            ADA = AdaBoostClassifier(base_estimator=DT,random_state=5)
            ADA.fit(X_train,y_train)
            Y_prea = ADA.predict(X_test)
            fn = confusion_matrix(y_test,Y_prea)
            FN = fn[1][0]
            metrix_T1.append(FN)
            a.append(one)
            b.append(two)
    bestt  = pd.DataFrame({'max_leaf_nodes':a ,'max_depth':b, 'type 2 error':metrix_T1 })
    bestt = bestt[bestt['type 2 error'] == bestt['type 2 error'].min()]
    if st.checkbox('click here to get best parameter'):
     st.dataframe(bestt.head())
    else:
     st.sidebar.write("")

    chi2, p = mcnemar(ary=Counfusion_metrix, exact=True)
    mcn = st.checkbox("Mc nemar test")
    if mcn:
         st.write('chi-squared:', chi2)
         st.write('p-value:', p)

    y_pred_proba_clf = ADA.predict_proba(X_test)
    y_pred_proba_clf =  y_pred_proba_clf[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_clf)
    r_auc = roc_auc_score(y_test, y_pred_proba_clf)


    # Seaborn's beautiful styling
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
    fig2=plt.figure(figsize=(25, 13))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='DT (AUROC = %0.3f)' % r_auc,marker='*')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend()
    # plt.show()

    ROC = st.checkbox("ROC CURVE")
    if ROC:
        st.pyplot(fig2)
        st.write("AUC:",r_auc)

GBC = st.sidebar.checkbox('GradientBoostingClassifier')
if GBC:
    GB = GradientBoostingClassifier(random_state=random)        
    GB.fit(X_train,y_train)
    Y_pre = GB.predict(X_test)
    Accuracy = accuracy_score(y_test,Y_pre)
    Precision = precision_score(y_test,Y_pre)
    Recall  = recall_score(y_test,Y_pre)
    F1_score = f1_score(y_test,Y_pre)
    Fbeta  = fbeta_score(y_test,Y_pre,average='macro', beta=0.5)
    Counfusion_metrix = confusion_matrix(y_test,Y_pre)
    Classification_report = classification_report(y_test,Y_pre)
    clsreport = print(Classification_report)
    figgg, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(Counfusion_metrix), annot=True, cmap="YlGnBu" ,fmt='g',ax=ax)
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        GB, X_train.values, y_train.values, X_test.values, y_test.values, 
        loss='0-1_loss',
        random_seed=random)
    
    bias = st.checkbox("Bias_variance_decomp")
    if bias:
        st.write('Average expected loss: %.3f' % avg_expected_loss)
        st.write('Average bias: %.3f' % avg_bias)
        st.write('Average variance: %.3f' % avg_var)
    report = st.checkbox("Classification_report")
    if report:
        st.text(Classification_report)
    metrix = st.checkbox(" Counfusion_metrix")
    if metrix:
        st.write(figgg)

    y_pred_proba_clf = GB.predict_proba(X_test)
    y_pred_proba_clf =  y_pred_proba_clf[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_clf)
    r_auc = roc_auc_score(y_test, y_pred_proba_clf)


    # Seaborn's beautiful styling
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
    fig2=plt.figure(figsize=(25, 13))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='DT (AUROC = %0.3f)' % r_auc,marker='*')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend()
    # plt.show()

    ROC = st.checkbox("ROC CURVE")
    if ROC:
        st.pyplot(fig2)
        st.write("AUC:",r_auc)

    chi2, p = mcnemar(ary=Counfusion_metrix, exact=True)
    mcn = st.checkbox("Mc nemar test")
    if mcn:
         st.write('chi-squared:', chi2)
         st.write('p-value:', p)