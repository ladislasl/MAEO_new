import streamlit as st
import tkinter as tk
from tkinter import filedialog 
import glob
import cv2
import numpy as np
import time
import csv
import pandas as pd

running = False
num_frames=1

st.title("MAEO's Elasmobranchii Recognition")
intro_title = st.empty()
intro_title.text('Please select the directory containing your videos.')
folder_path = st.text_input('Indicate the folder/local path containing your videos (ex: C:/User/videos):', value='')
video_list = glob.glob(str(folder_path)+"/*.mp4")
st.write('Number of videos found:', len(video_list))
displayer = st.empty()
radio = st.empty()

with st.sidebar:
    """
    ## :floppy_disk: Parameters:

    """
    fps_rate = st.select_slider('Select the number of analyzed frames per second:', options=[1,2,3,5,6,10,15,30])


intro_title.empty()  

if video_list:
    with st.sidebar:
        """
        ## :camera: Analyze:

        """
        dict = {path.split("/")[-1] : path for path in video_list}
        select_video = st.selectbox("Select a video:", options=dict.keys())
        process_video = st.button('Process video')
        status_text = st.sidebar.empty()
        progress_bar = st.sidebar.progress(0)
    

    if select_video:
        try:
            current_video = cv2.VideoCapture(dict[select_video])
            num_frames = int(current_video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(current_video.get(cv2.CAP_PROP_FPS)) 
            with displayer.container():
                frame_counter = st.slider('Select the frame to show:', min_value=0, max_value=num_frames-1, value=0, step=max(1,fps_rate))
                frame_txt = st.empty()
                stframe = st.empty()
                
            
            current_video.set(1,frame_counter)
            ret, showed_frame = current_video.read()
            frm = cv2.cvtColor(showed_frame, cv2.COLOR_BGR2RGB)
            ms = current_video.get(cv2.CAP_PROP_POS_MSEC)
            frame_txt = st.text(time.strftime('%M:%S:{}'.format(int(ms%1000)), time.gmtime(ms/1000)))
            stframe = st.image(frm, width = 720)
        except:
            st.warning('We cannot determine number of frames and FPS! Please check video format.')
        

        if process_video:
            with st.spinner("Video being analyzed... Please wait. It may take several minutes."):
                #NEURAL NETWORK HERE
                for i in range(101):
                    status_text.text("%i%% Complete" % i)
                    progress_bar.progress(i)
                    time.sleep(0.01)
            
            progress_bar.empty()
            st.success('Done!')
            df = pd.read_csv('../csv/example.csv', delimiter=';')
            st.dataframe(df)
            obs_list = [str(i)+' : '+str(df.loc[i,'Observation'])+' ('+str(df.loc[i,'Confidence'])+') ['+str(df.loc[i,'Frame'])+']' for i in df.index]
            with st.sidebar:
                """
                ## :fish: Visualizations:"""
                st.radio("You can visualize observations:",obs_list)
            process_video = False

    else:
        st.write('Please indicate a directory containing MP4 files.')
