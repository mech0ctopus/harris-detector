# -*- coding: utf-8 -*-
"""
Stitch images using Harris Detector and SIFT.
"""
from harris_detector import harris_detector
from descriptor_utils import blending, get_homography, run_sift
import cv2

def image_stitch(img_paths,first_image_id=0,second_image_id=1,output_path='Composite.png'):
    '''Stitch two images using Harris Detector and SIFT.'''
    #Intialize variables
    keypoints={}  
    descriptors={}  
    #Get keypoints and SIFT descriptors for each corner in each image
    for image_idx, img_path in enumerate(img_paths):
        #Get corner locations
        corner_locations=harris_detector(img_path,threshold=0.2496,show=False)
        #Create descriptors
        keypoints[image_idx], descriptors[image_idx]=run_sift(img_path,
                                                              corner_locations,
                                                              size=3,
                                                              use_sift_detect=False)
    
    #Get homography matrix
    H=get_homography(descriptors[first_image_id],keypoints[first_image_id],  
                     descriptors[second_image_id],keypoints[second_image_id])
    
    #Load images
    img1 = cv2.imread(img_paths[first_image_id])
    img2 = cv2.imread(img_paths[second_image_id])
    #Create composite image
    result=blending(img1,img2,H)
    #Save composite image
    cv2.imwrite(output_path, result)
    
if __name__=='__main__':
    img_paths=[r"octopus\MiddleLeft.PNG",
               r"octopus\MiddleMiddle.PNG",
               r"octopus\BottomMiddle.PNG",
               r"octopus\MiddleRight.PNG",
               r"Composite1.PNG",
               r"Composite2.PNG"]

    #Build panoramic image
    image_stitch(img_paths[:3],first_image_id=0,second_image_id=1,output_path=r"Composite1.png")
    image_stitch(img_paths[:4],first_image_id=2,second_image_id=3,output_path=r"Composite2.png")
    image_stitch(img_paths,first_image_id=4,second_image_id=5,output_path=r"Composite3.png")

