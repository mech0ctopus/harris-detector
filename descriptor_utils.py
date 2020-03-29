# -*- coding: utf-8 -*-
"""
Keypoint and Descriptor Utilities.
"""
import cv2
import numpy as np

def corners_to_keypoints(corners,size=30):
    '''Converts x,y pixel positions to cv2.KeyPoints.'''
    keypoints=[]
    for corner in corners:
        kp=cv2.KeyPoint(corner[1],corner[0],_size=size)
        keypoints.append(kp)
        
    return keypoints

def run_sift(img_path,corner_locations,size,use_sift_detect=False):
    '''Run SIFT to build list of descriptors.'''
    #Instantiate SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    #Convert corner locations to keypoint objects
    if use_sift_detect==False:
        keypoints=corners_to_keypoints(corner_locations,size=size)
    #Load image into grayscale
    img=cv2.imread(img_path,0)
    #Compute descriptors
    if use_sift_detect==False:
        keypoints, descriptors = sift.compute(img,keypoints)
    else:
        keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def get_homography(descriptor_1,keypoints_1,descriptor_2,keypoints_2):
    '''Get descriptor matches from two images.''' 
    #Try matching image descriptors  
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(descriptor_1, descriptor_2, k=2)
    good_points = []
    good_matches=[]
    for m1, m2 in raw_matches:
        if m1.distance < 0.85 * m2.distance:
            good_points.append((m1.trainIdx, m1.queryIdx))
            good_matches.append([m1])

    if len(good_points) > 8:
        image1_kp = np.float32(
            [keypoints_1[i].pt for (_, i) in good_points])
        image2_kp = np.float32(
            [keypoints_2[i].pt for (i, _) in good_points])
    
    (H, status) = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5)
    
    return H

def draw_matches(matches, img_path1, keypoints1, img_path2, keypoints2):
    '''Draw descriptor and keypoint matches.'''
    #-- Draw matches
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), 
                            img1.shape[1]+img2.shape[1], 3), 
                            dtype=np.uint8)
    cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)
    #-- Show detected matches
    cv2.imshow('Matches', img_matches)
    

def create_mask(img1,img2,version):
    '''Masking function for blending.'''
    smoothing_window_size=188
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 +width_img2
    offset = int(smoothing_window_size / 2)
    barrier = img1.shape[1] - int(smoothing_window_size / 2)
    mask = np.zeros((height_panorama, width_panorama))
    if version== 'left_image':
        mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
        mask[:, :barrier - offset] = 1
    else:
        mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
        mask[:, barrier + offset:] = 1
    return cv2.merge([mask, mask, mask])

def blending(img1,img2,H):
    '''Blend two images into single composite.'''
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 +width_img2

    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 = create_mask(img1,img2,version='left_image')
    panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
    panorama1 *= mask1
    mask2 = create_mask(img1,img2,version='right_image')
    panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
    result=panorama1+panorama2

    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    return final_result