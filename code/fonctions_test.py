        
        
        
        
video = dict[select_video]
directory = "save/"+select_video[7:-4]+"/"
if not os.path.exists(directory):
    os.makedirs(directory)

vidcap = cv2.VideoCapture(select_video)
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("frames/frame%d.jpg" % count, image)      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    run(weights='yolov5/best_60.pt', imgsz=416, conf_thres=0.6, source="frames/frame%d.jpg" % count, name="frames_vid2_traced/frame%d.jpg" % count+'_results', save_txt=True, save_conf=True)
    count += 1
