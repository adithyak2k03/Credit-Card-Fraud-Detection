from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

import pickle
from tensorflow.keras.models import load_model


def main():
    st.title("Credit Card Fraud Detection")

if __name__ == '__main__':
    main()
st.sidebar.title("Choose Dataset")
Dataset = st.sidebar.radio("",("Imbalanced","Upsampled","Downsampled"))


if Dataset=='Imbalanced':
    df=pd.read_csv('creditcard_prep.csv')
    path='imbalanced/'
elif Dataset=='Upsampled':
    df=pd.read_csv('creditcard_up.csv')
    path='upsampled/'
elif Dataset=='Downsampled':
    df=pd.read_csv('creditcard_down.csv')
    path='downsampled/'

# st.write(path+Dataset)

if st.sidebar.checkbox("Display data", False):
    st.subheader("European Credit Card Fraud dataset")
    st.table(df.head())
    

st.sidebar.title("Choose Classifier")
    



@st.cache_data(persist=True)
def split(df):
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    # x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)
    
    x_test = pickle.load(open(path+'X_test.pkl', 'rb'))
    y_test = pickle.load(open(path+'y_test.pkl', 'rb'))
    x_train = pickle.load(open(path+'X_train.pkl', 'rb'))
    y_train = pickle.load(open(path+'y_train.pkl', 'rb'))
    
    
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = split(df)
# x_train.shape

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)


def eval_metrics(y_pred):
    st.write("Accuracy: ", accuracy_score(y_test, y_pred).round(3))
    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(3))
    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(3)) 
    st.write("F1-Score: ", f1_score(y_test, y_pred, average='macro').round(3))
    
def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=   class_names)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()
        
def NN_plot_metrics(metrics_list,):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        # plot_confusion_matrix(model, x_test, y_test, display_labels=   class_names)
        st.pyplot()
    if "Accuracy Curve" in metrics_list:
        st.subheader("Accuracy Curve")
        plt.plot(history['loss'], label='Training loss')
        plt.plot(history['val_loss'], label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.show()
        st.pyplot()
    if "Loss Curve" in metrics_list:
        st.subheader("Loss Curve")
        
        st.pyplot()
class_names = ["Fraud", "Not Fraud"]
count=0
classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest","ANN","XG-Boost","CNN"))


def metric_sel():
    global count
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key = count)
    count=count+1
    # 157892
    return metrics

def NN_metric_sel():
    global count
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "Accuracy Curve", "Loss Curve"), key = count)
    count=count+1
    return metrics



if classifier == "Support Vector Machine":
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear","poly","sigmoid"), key="kernel") 
    metrics = metric_sel()
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Support Vector Machine (SVM) results")
        # model = SVC(C=C, kernel=kernel, gamma=gamma)
        if kernel=='rbf':
            model = pickle.load(open(path+'svc_rbf.pkl', 'rb'))
        elif kernel=='linear':
            model = pickle.load(open(path+'svc_lin.pkl', 'rb'))
        elif kernel=='poly':
            model = pickle.load(open(path+'svc_poly.pkl', 'rb'))
        elif kernel=='sigmoid':
            model = pickle.load(open(path+'svc_sig.pkl', 'rb'))
        


        y_pred = model.predict(x_test)
        
        eval_metrics(y_pred)
        
        plot_metrics(metrics)
st.set_option('deprecation.showPyplotGlobalUse', False)


if classifier == "Logistic Regression":

    metrics = metric_sel()

    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        # model = LogisticRegression(random_state=0)
        model = pickle.load(open(path+'LR.pkl', 'rb'))
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        eval_metrics(y_pred)
        plot_metrics(metrics)

if classifier == "K-Nearest Neighbors":

    metrics = metric_sel()

    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("K-Nearest Neighbors Results")
        
        model = pickle.load(open(path+'knn.pkl', 'rb'))
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        
        eval_metrics(y_pred)
        
        plot_metrics(metrics)
        
if classifier == "Decision Tree":

    metrics = metric_sel()

    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Decision Tree Results")
        # model = LogisticRegression(random_state=0)
        model = pickle.load(open(path+'dt.pkl', 'rb'))
        model.fit(x_train, y_train)

        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        eval_metrics(y_pred)
        
        plot_metrics(metrics)
         
if classifier == "Random Forest":

    metrics = metric_sel()
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        # model = LogisticRegression(random_state=0)
        model = pickle.load(open(path+'rf.pkl', 'rb'))
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        
        eval_metrics(y_pred)
        
        plot_metrics(metrics)

if classifier == "XG-Boost":

    metrics = metric_sel()

    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("XG Boost Results")
        
        model = pickle.load(open(path+'xgb.pkl', 'rb'))
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        
        eval_metrics(y_pred)
        
        plot_metrics(metrics)

if classifier == "ANN":

    metrics = metric_sel()

    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("ANN Results")

        model = load_model(path+'ANN_model.h5')
        # accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        for i in range(len(y_pred)):
            y_pred[i]=y_pred[i].round()
        
        for i in range(len(y_pred)):
            y_pred[i]=y_pred[i].round()
        
        eval_metrics(y_pred)
        
        plot_metrics(metrics)

# classifier_LR.predict(sc.transform(test_x))

if classifier == "CNN":
    
    # metrics = metric_sel()
    # NN_metrics = NN_metric_sel()
    
    num_layers = st.sidebar.radio("No. of Layers", ("14", "17","20"), key="Num layers") 
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("CNN Results")
        # model = SVC(C=C, kernel=kernel, gamma=gamma)
        if num_layers=='14':
            model = load_model(path+'CNN_14_layers.h5')
        elif num_layers=='17':
            model = load_model(path+'CNN_17_layers.h5')
        elif num_layers=='20':
            model = load_model(path+'CNN_20_layers.h5')
    

        y_pred = model.predict(x_test)
        for i in range(len(y_pred)):
            y_pred[i]=y_pred[i].round()
            
        eval_metrics(y_pred)
        
        # NN_plot_metrics(NN_metrics)    
        # plot_metrics(metrics)






df1=pd.read_csv('temp_report.csv',index_col=0)

if Dataset=='Imbalanced':
    df1=pd.read_csv('reports/'+path[:-1]+'.csv',index_col=0)
    path='imbalanced/'
elif Dataset=='Upsampled':
    df1=pd.read_csv('reports/'+path[:-1]+'.csv',index_col=0)
    path='upsampled/'
elif Dataset=='Downsampled':
    df1=pd.read_csv('reports/'+path[:-1]+'.csv',index_col=0)
    path='downsampled/'


if st.sidebar.checkbox("Display Report", False):
    st.subheader("Report")
    # st.write(df1)
    st.table(df1)
    
pred_val_or_not= st.sidebar.checkbox('Predict a value ')

# st.write(str(x_train[0]))
if pred_val_or_not:
    l=x_train[[0]]
    l=str(l).replace("\n","")
    l=str(l).replace("  "," ")
    l=str(l).replace(" ",",")
    l=str(l).replace("[ ","")
    l=str(l).replace("[,","[")
    l=str(l).replace(",,",",")

    # st.write(l)
    test_x = st.text_input('Enter data to predict', l)
    
    import ast
    
    #just for sample we are taking knn model
    model = pickle.load(open('knn.pkl', 'rb'))
    model.fit(x_train, y_train)
    test_x = ast.literal_eval(test_x)
    predicted_test_x= model.predict(sc.transform(test_x))

    st.write('The  prediction is',predicted_test_x)


from sklearn.preprocessing import StandardScaler
userinp = st.file_uploader("Upload file", type={"csv", "txt"})
if userinp is not None:
    pred_df = pd.read_csv(userinp)

    dataset2 = pred_df
    # Normalizace the Amount between -1,1
    dataset2['normalizedAmount'] = StandardScaler().fit_transform(dataset2['Amount'].values.reshape(-1,1))

    a=dataset2['Class']
    b=dataset2['normalizedAmount']

    dataset2 = dataset2.drop(columns = ['Class','normalizedAmount'])

    dataset2['normalizedAmount']=b
    dataset2['Class']=a

    pred_df1 = dataset2.drop(columns = ['Amount','Time'])
    st.write(pred_df1.head())
    x = pred_df1.iloc[:,:-1]
    y = pred_df1.iloc[:,-1]
    model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    model.fit(x_train, y_train)
    predop = model.predict(x)
    inds = predop==1
    malicious = pred_df[inds]
    st.subheader("Identified as malicious")
    st.write(malicious.head())
