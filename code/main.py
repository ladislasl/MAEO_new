
import streamlit as st
import tkinter as tk
from tkinter import filedialog
import glob
import cv2
import numpy as np
import time
import csv
import pandas as pd
import torch
from yolov5.detect import *
import os

running = False
num_frames=1

st.title("MAEO's Elasmobranchii Recognition")
intro_title = st.empty()
intro_title.text('Please select the directory containing your videos.')
folder_path = st.text_input('Indicate the folder/local path containing your videos (ex: C:/User/videos):', value='C:/Users/ladis/OneDrive/Bureau/MAEO/code/videos')
video_list = glob.glob(str(folder_path)+"/*.mp4")
st.write('Number of videos found:', len(video_list))
displayer1 = st.empty()
radio = st.empty()
displayer2 = st.empty()
video_analized = False

@st.cache()
def load_csv():
    try:
        return output_df
    except:
        return pd.read_csv('csv/example.csv', delimiter=';')

def processFrame(weights,imgsz,conf_thres,source):
    run(weights=weights, imgsz=imgsz, conf_thres=conf_thres, source=source)

with st.sidebar:
    """
    ## :floppy_disk: Parameters:

    """
    fps_rate = st.select_slider('Select the number of analyzed frames per second:', options=[1,2,3,5,6,10,15,30])


intro_title.empty()
with st.sidebar:
    """
    ## :camera: Analyze:

    """
    dict = {path.split("/")[-1] : path for path in video_list}
    select_video = st.selectbox("Select a video:", options=dict.keys())
    process_video = st.button('Process video')
    status_text = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)

if not video_list:
    st.write('Please indicate a directory containing MP4 files.')
    

if select_video:
    try:
        current_video1 = cv2.VideoCapture(dict[select_video])

        num_frames = int(current_video1.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(current_video1.get(cv2.CAP_PROP_FPS)) 
        with displayer1.container():
            frame_counter = st.slider('Select the frame to show:', min_value=0, max_value=num_frames-1, value=0, step=max(1,fps_rate))
            frame_txt1 = st.empty()
            stframe1 = st.empty()
        with displayer2.container():
            #frame_counter = st.slider('Select the frame to show:', min_value=0, max_value=num_frames-1, value=0, step=max(1,fps_rate))
            frame_txt2 = st.empty()
            stframe2 = st.empty()
            
        current_video1.set(1,frame_counter)
        ret, showed_frame = current_video1.read()
        frm1 = cv2.cvtColor(showed_frame, cv2.COLOR_BGR2RGB)
        ms1 = current_video1.get(cv2.CAP_PROP_POS_MSEC)
        frame_txt1 = st.text(time.strftime('%M:%S:{}'.format(int(ms1%1000)), time.gmtime(ms1/1000)))
        stframe1 = st.image(frm1, width = 720)

    except:
        st.warning('We cannot determine number of frames and FPS! Please check video format.')
        
        
if process_video:
    with st.spinner("Video being analyzed... Please wait. It may take several minutes."):
        #NEURAL NETWORK HERE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        video = dict[select_video]
        print(select_video)
        directory = "save/"+select_video[7:-4]+"/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        run(weights='yolov5/best_60.pt', imgsz=416, conf_thres=0.6, source=select_video, name=select_video[7:-4]+'_results', save_txt=True, save_conf=True)
        #NEURAL NETWORK HERE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #output_df = pd.read_csv('csv/example.csv', delimiter=';')
    #data = [(3, 'a'), (2, 'b'), (1, 'c'), (0, 'd')] #A CHANGER
    #progress_bar.empty()
    st.success('Done!')
    video_analized = True
    process_video = False

#C:\Users\ladis\OneDrive\Bureau\MAEO\code\yolov5\runs\detect\une-enorme-raie-pastenague_results\une-enorme-raie-pastenague.mp4
current_video2 = cv2.VideoCapture("yolov5/runs/detect/une_enorme_raie_pastenague_results/une_enorme_raie_pastenague.mp4")
with displayer2.container():
            frame_counter2 = st.slider('Select the frame to show for vid result:', min_value=0, max_value=num_frames-1, value=0, step=max(1,fps_rate))
            frame_txt2 = st.empty()
            stframe2 = st.empty()
current_video2.set(1,frame_counter2)
ret, showed_frame = current_video2.read()
frm2 = cv2.cvtColor(showed_frame, cv2.COLOR_BGR2RGB)
ms2 = current_video2.get(cv2.CAP_PROP_POS_MSEC)
frame_txt2 = st.text(time.strftime('%M:%S:{}'.format(int(ms2%1000)), time.gmtime(ms2/1000)))
stframe2 = st.image(frm2, width = 720)




#df = load_csv()
#st.dataframe(df)
#obs_list = [str(i)+' : '+str(df.loc[i,'Observation'])+' ('+str(df.loc[i,'Confidence'])+') ['+str(df.loc[i,'Frame'])+']' for i in df.index]
#with st.sidebar:
#    """
#    ## :fish: Visualizations:"""
#    frm_choice = st.radio("You can visualize observations:", obs_list)
#current_video2.set(1,int(frm_choice.split('[')[-1][:-1]))        
#ret, showed_frame = current_video2.read()
#frm2 = cv2.cvtColor(showed_frame, cv2.COLOR_BGR2RGB)
#ms2 = current_video2.get(cv2.CAP_PROP_POS_MSEC)
#frame_txt2 = st.text(time.strftime('%M:%S:{}'.format(int(ms2%1000)), time.gmtime(ms2/1000)))
#stframe2 = st.image(frm2, width = 720)
    

#C:/Yann/Mines/Césure/Stages/ARBRE/MAEO/code/videos