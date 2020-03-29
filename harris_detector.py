# Craig D. Miller
# cdmiller@wpi.edu
# Advanced Robot Navigation
#
# Harris Detector

import cv2
from harris_utils import compute_G
from harris_utils import corner_assignment, load_image
from harris_utils import get_derivative, compute_response

def harris_detector(image_path,threshold=200,show=True):
    '''Returns locations of all detected corners (pixel-by-pixel).'''
    if type(image_path) != str:
        #Assume image_path was given as img
        img_rgb=image_path
        height,width=img_rgb.shape[:-1]
        img=cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    else:
        #Load image from path
        img,width,height=load_image(image_path,gray=True,show=False)
        img_rgb,_,_=load_image(image_path,gray=False,show=False)
    
    #Step 1: Get fx and fy derivatives
    fx=get_derivative(img,height,width,direction='x',show=False)
    fy=get_derivative(img,height,width,direction='y',show=False)
    
    #Step 2: Compute overall G, Structure Matrix
    G=compute_G(fx,fy)

    #Step 3: Calculate eigenvalues and get Harris value
    harris_response=compute_response(G,threshold)
    
    #Step 4: Assign corners
    image_overlay, corner_locations=corner_assignment(img_rgb,harris_response)
    
    if show==True:
        #Show corners overlayed on original RGB
        cv2.imshow('Detected Corners',image_overlay)
    
    return corner_locations

if __name__=='__main__':
    img_path=r"octopus\MiddleRight.PNG"
    corner_locations=harris_detector(img_path,threshold=.245)
    print(len(corner_locations))
    
        