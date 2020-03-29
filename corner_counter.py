# -*- coding: utf-8 -*-
"""
Counts and displays numbers of corners found
in each frame of a video (28FPS) using Harris Detector. 
Generates and displays new video at 10 FPS.
"""
    
import cv2
from harris_detector import harris_detector

def find_corners(img):
    '''Finds corners in image using harris detector. Writes number of 
    corners on image and saves new image.'''
    #Find number of corners in image
    corner_locations=harris_detector(img,threshold=0.2496,show=False)
    num_corners=len(corner_locations)

    #Write text on image
    cv2.putText(img,f'Corners Detected: {num_corners}', 
                (10,1060), #Bottom left corner of text
                cv2.FONT_HERSHEY_SIMPLEX, #Font
                2, #Font Scale
                (0,0,255), #Font Color
                4) #Line Type
    return img
    
def read_write_video(filename,out_FPS=10,output_filename=r'output.avi'):
    '''Reads video and writes number of detected corners, then saves new
    video.'''
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(filename)

    #Get resolution
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    # Define the codec and create VideoWriter object.
    out = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc('M','J','P','G'), 
                          out_FPS, (frame_width,frame_height))
    
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            #Write corner count text to frame
            corner_frame=find_corners(frame)
            # Write the frame into the file 'output.avi'
            out.write(corner_frame)
        
        # Break the loop
        else: 
            break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

if __name__=='__main__':
    vid_path=r"videos\crazy_video.mov"
    read_write_video(vid_path)