import cv2
import numpy as np

video = cv2.VideoCapture('LineVideo.mp4')
car_cascade = cv2.CascadeClassifier('cars.xml')


def ROI(frame_org):
    polygons = np.array([[(572,441),(679,446),(1103,666),(300,681)]])
    mask = np.zeros_like(frame_org)
    fill = cv2.fillPoly(mask,polygons,255)
    mask_inv = cv2.bitwise_not(fill)
    masked_image = cv2.bitwise_and(mask_inv,frame_org)
    return masked_image

while True:
    ret,frame = video.read()
    if not ret:
        video = cv2.VideoCapture('LineVideo.mp4')
        continue

    reg = frame[440:720,300:900]

    cvthsv = cv2.cvtColor(reg,cv2.COLOR_BGR2HSV)

    frame_org = cv2.GaussianBlur(reg,(5,5),0)
    hsv = cv2.cvtColor(frame_org,cv2.COLOR_BGR2HLS)
    L = hsv[:,:,1]
    L_max, L_mean = np.max(L), np.mean(L)
    S = hsv[:,:,2]
    S_max, S_mean = np.max(S), np.mean(S)

    yellow_lower = np.array([5, 75, 127])
    yellow_upper = np.array([48, 255,255])
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    edges = cv2.Canny(mask_yellow, 75, 150)

    L_adapt_white =  max(160, int(L_max * 0.8),int(L_mean * 1.25))
    hls_low_white = np.array((0, L_adapt_white,  0))
    hls_high_white = np.array((255, 255, 255))
    hls_white = cv2.inRange(hsv, hls_low_white, hls_high_white)
    edges1 = cv2.Canny(hls_white,75, 150)

    comboedges = edges + edges1

    lines = cv2.HoughLinesP(comboedges,1,np.pi/180,50,maxLineGap=15)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(reg,(x1,y1),(x2,y2),(0,255,0),5)
    
    cv2.imshow('Video_org',frame)
    cropped_video = ROI
    startvideo = cropped_video(reg)
    cv2.imshow('Roi',startvideo)


    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()