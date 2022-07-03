import streamlit as st 
import soundfile as sf
import numpy as np
import librosa
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from statistics import mode
    

st.title('Bird Species Classification')
r_forest=8


df = pd.read_csv('archive/Bird_1Sec_Mfcc.csv')
x = df.iloc[:,2:42]
y = df.iloc[:,42]
x.shape

x_train, x_test, y_train, y_test = train_test_split( x , y ,test_size=30 )
clf = RandomForestClassifier()
clf.fit( x_train , y_train )
st.success("Model loaded successfully!")



def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Image Loaded")


def extract_mfcc(filename,start,duration,n_mfcc):
    y, sr = librosa.load(filename, duration=duration, offset=start)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    return mfcc

uploaded_file=st.file_uploader('Choose Audio file',type=['wav','ogg'])


if uploaded_file is not None:
    save_uploadedfile(uploaded_file)
    filename=os.path.join("tempDir",uploaded_file.name)
    features=[]
    data, sr = librosa.load(filename)
    l = librosa.get_duration(y=data, sr=sr)

    for i in range(0,int(l)):
        features.append( extract_mfcc(filename,i,1,40) )
    
    y_p = []
    for i in features:
        u = clf.predict([i])
        y_p.append(u[0])
    
    dg1 = mode(y_p)
    st.write(dg1)
