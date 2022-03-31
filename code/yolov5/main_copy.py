
from unittest import result
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
#from yolov5.detect import *
import os
import csv


cuda0 = torch.device('cuda:0')


def traitement_csv(df):

    df1 = pd.DataFrame(columns=['Observation','Frame','Confidence'])
    df3 = pd.DataFrame({'Observation': df.loc[0,"Observation"], 'Frame': df.loc[0,"Frame"], 'Confidence': df.loc[0,"Confidence"]}, index=[0])
    df1 = pd.concat([df1, df3], ignore_index = True, axis = 0)
    k=0
    
    while k<len(df)-2:
        
        if df.loc[k,"Observation"]!=df.loc[k+1,"Observation"] or df.loc[k,"Frame"]+1 != df.loc[k+1,"Frame"]:
            df3 = pd.DataFrame({'Observation': df.loc[k+1,"Observation"], 'Frame': df.loc[k+1,"Frame"], 'Confidence': df.loc[k+1,"Confidence"]}, index=[0])
            df1 = pd.concat([df1, df3], ignore_index = True, axis = 0)
            k+=1
        else :
            while k<len(df)-2 and df.loc[k,"Observation"]==df.loc[k+1,"Observation"] and df.loc[k,"Frame"]+1 == df.loc[k+1,"Frame"]:
                k+=1
                print(k)
    return df1




running = False
num_frames=1
st.title("MAEO's Elasmobranchii Recognition")
intro_title = st.empty()
intro_title.text('Please select the directory containing your videos.')
folder_path = st.text_input('Indicate the folder/local path containing your videos (ex: C:/User/videos):', value='C:/Users/ladis/OneDrive/Bureau/MAEO/code/videos')
video_list = glob.glob(str(folder_path)+"/*.mp4")
print(video_list)
st.write('Number of videos found:', len(video_list))
displayer1 = st.empty()
radio = st.empty()
displayer2 = st.empty()
video_analized = False
result_path= folder_path[0:-12] + "/code/yolov5/runs/detect"

result_list=glob.glob(str(result_path)+"/*")
#print(result_list)


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
    dict_results= {path.split("/")[-1] : path for path in result_list}
    select_video = st.selectbox("Select a video:", options=dict.keys())
    process_video = st.button('Process video')
    my_bar = st.progress(0)
    status_text = st.sidebar.empty()
    video_choisie=st.selectbox("results", options=dict_results.keys())

if not video_list:
    st.write('Please indicate a directory containing MP4 files.')
#######################################################################################################################################

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    # Run inference
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    percent_complete= 0
    list_labels=[]
    num=0
    for path, im, im0s, vid_cap, s in dataset:
        percent_complete+=1/(length+0.1)
        my_bar.progress(percent_complete)
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        #with open(txt_path + '.csv', 'a') as f:
                        #    f.write()

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        list_labels.append([label, num])
                        

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        num+=1
    
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(list_labels)

    with open(save_path[:-4] + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Observation','Frame','Confidence'])
        for label in list_labels:
            writer.writerow([label[0][:-4],label[1],label[0][-4:]])
            
        
    return save_path


    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt
()

if process_video:
    with st.spinner("Video being analyzed... Please wait. It may take several minutes."):
        #NEURAL NETWORK HERE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        video = dict[select_video]
        print(select_video)
        directory = "save/"+select_video[7:-4]+"/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        bon_path=run(weights='best_60.pt', imgsz=416, conf_thres=0.6, source="C:/Users/ladis/OneDrive/Bureau/MAEO/code/"+select_video, name=select_video[7:-4]+'_results', save_txt=True, save_conf=True)
        #NEURAL NETWORK HERE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #output_df = pd.read_csv('csv/example.csv', delimiter=';')
    #data = [(3, 'a'), (2, 'b'), (1, 'c'), (0, 'd')] #A CHANGER
    #progress_bar.empty()
    st.success('Done!')
    video_analized = True
    #process_video = False
    print(bon_path)
    bon_path="C:/Users/ladis/OneDrive/Bureau/MAEO/code/yolov5/"+ bon_path.replace("\\", "/")
    print("BOOONNPATTH",bon_path)
    process_video=False
 


if video_choisie:
    try:

        liste_vid_interne = glob.glob(str(dict_results[video_choisie].replace("\\", "/"))+"/*.mp4")
        vid= liste_vid_interne[0].replace("\\", "/")
        current_video1 = cv2.VideoCapture(vid)
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
        


    




liste_vid_interne = glob.glob(str(dict_results[video_choisie].replace("\\", "/"))+"/*.csv")
vid= liste_vid_interne[0].replace("\\", "/")
        
df = pd.read_csv(vid, delimiter=',')
df = traitement_csv(df)
#st.dataframe(df)
obs_list = [str(i)+' : '+str(df.loc[i,'Observation'])+' ('+str(df.loc[i,'Confidence'])+') ['+str(df.loc[i,'Frame'])+']' for i in df.index]
    
if video_choisie:
    with st.sidebar:
        """
        ## :fish: Visualizations:"""
        
        frm_choice = st.radio("You can visualize observations:", obs_list)
current_video1.set(1,int(frm_choice.split('[')[-1][:-1]))        
ret, showed_frame = current_video1.read()
frm2 = cv2.cvtColor(showed_frame, cv2.COLOR_BGR2RGB)
ms2 = current_video1.get(cv2.CAP_PROP_POS_MSEC)
frame_txt2 = st.text(time.strftime('%M:%S:{}'.format(int(ms2%1000)), time.gmtime(ms2/1000)))
stframe2 = st.image(frm2, width = 720)
    

#C:/Yann/Mines/Césure/Stages/ARBRE/MAEO/code/videos


