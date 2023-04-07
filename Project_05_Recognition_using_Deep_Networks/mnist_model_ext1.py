"""
Dev Vaibhav
Spring 2023 CS 5330
Project 5: Recognition using Deep Networks
Extension 1: MNIST digit recognition from live feed 
"""

import cv2
import imutils
import sys
import torch
import matplotlib.pyplot as plt


import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from helper import ThresholdTransform, MyNetwork
import os
from PIL import Image

# Function to detect figits from live video
# main function code
def main():

    # Live video code based on Src: https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

    network = MyNetwork()
    network.load_state_dict(torch.load("./results/model_1A_1E.pth"))
    network.eval()

    # A series of operations to get the desired input in the network
    convert_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), #Converts [0,255] to [0,1]
        torchvision.transforms.Grayscale(),
        # torchvision.transforms.CenterCrop((350,200)),
        torchvision.transforms.Resize((28,28), antialias=True),
        torchvision.transforms.RandomInvert(p=1.0),
        ThresholdTransform(thr_255=170),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]
    )
        # define a video capture object
    vid = cv2.VideoCapture(1)

    #Show live video using matplotlib: https://stackoverflow.com/questions/44598124/update-frame-in-matplotlib-with-live-camera-preview

    # Interactive mode on for matplotlib pyplot module
    plt.ion()

    #Infinite loop to show/ process the live video
    while(True):
        print(" ################################### LOOP STARTED ###################################")
        # Capture the video frame by frame
        ret, frame_color = vid.read()

        # Resizing to 500x500
        frame_color = cv2.resize(frame_color, (500,500)) #cv::Mat
        # Changing order from BGR ti RGB so that it can be used by PIL
        frame_rgb = cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB) # cv::Mat
        
        # Grayscale image for processing
        frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        # Gaussian blur applied on the grayscale image
        blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        # Canny edge detection applied on the blurred image
        edged = cv2.Canny(blurred, 50, 200, 255)

        

        # find contours in the edge map, then sort them by their
        # size in descending order
        #Src: https://pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        cv2.drawContours(frame_gray, cnts, -1, (0, 255,0), 3)
        
        print("Number of contours detected: ", len(cnts))
        pbibb = 20 #Pixel buffer in bb. Used to move AABB corners by this much pixel value
        
        roi_array = [] #Array of images containg the region of interes (digits) in the image
        tl_array = [] #Co-ord of top ledt corner of AABB
        
        # loop over the contours
        for c in cnts:
            # Find the area of  the contour
            area = cv2.contourArea(c)
            if area > 300 and area < 3000: #Pick contour within this area range only
                # compute the center of the contour
                # Finding center of contours using moments
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                cv2.drawContours(frame_gray, [c], -1, (0, 255, 0), 2)
                cv2.circle(frame_gray, (cX, cY), 7, (255, 255, 255), -1)
                # cv2.putText(frame_gray,str(area), (cX - 20, cY - 20),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # Find corners of the AABB
                x,y,w,h = cv2.boundingRect(c)
                # draw the bounding rectangle | https://stackoverflow.com/questions/21104664/extract-all-bounding-boxes-using-opencv-python
                # print(frame_color.shape[0])
                
                # Top left corner coordinate of the AABB accounting for not too tight packing
                tl_coord = (cX-w//2 - pbibb, cY-h//2 - pbibb)
                # Ignoring AABB which is very close to the edge
                if (tl_coord[0] < pbibb or tl_coord[0] > frame_color.shape[0] - pbibb or tl_coord[1] < pbibb or tl_coord[1] > frame_color.shape[1] - pbibb) != 1:
                    # frame_gray = cv2.rectangle(frame_gray,(x-pbibb,y-pbibb),(x+max(w,h)+(pbibb//4),y+max(w,h)+(pbibb//4)),(0,0,255),1)
                    # roi = frame_color[y-pbibb:y+max(w,h)+(pbibb//4),x-pbibb:x+max(w,h)+(pbibb//4)]
                    new_width = w + 2*pbibb
                    new_height = h + 2*pbibb
                    # print(y-pbibb, " ", y+max(w,h)+pbibb, " ", x-pbibb, " ", x+max(w,h)+pbibb)
                    print(tl_coord[0], " ", new_width, " ", tl_coord[1], " ", new_height)

                    # Extracting region of interest from the image
                    roi = frame_color[tl_coord[1]:tl_coord[1] + new_height,tl_coord[0]:tl_coord[0] + new_width]
                    frame_color = cv2.rectangle(frame_color,tl_coord,(tl_coord[0] + new_width,tl_coord[1] + new_height),(0,0,255),1)

                    # Storing the roi and top left coordinate for future use
                    roi_array.append(roi)
                    tl_array.append(tl_coord)

                # cv2.waitKey(5000)


        
        

        #Modes: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        # Src: https://www.geeksforgeeks.org/convert-opencv-image-to-pil-image-in-python/
        
        

        print("Number of ROI: ", len(roi_array))
        # Only process if at least one roi is detected
        if (len(roi_array) != 0):
            batch = torch.zeros(len(roi_array),1,28,28)#,dtype=float)
            # Process for all roi
            for i in range(len(roi_array)):
                # cv2.imshow('ROI image: ' + str(i) , roi_array[i])
                # im3 = ax3.imshow(roi_array[i]) #Matplot lib expects input in RGB order but openCV is BGR
                # im3.set_data(roi_array[i])
                
                # CV image to PIL image
                frame_rgb_PIL = Image.fromarray(roi_array[i], mode="RGB") #Convert the cv::Mat to PIL image which is fine
                # PIL to tensor conversion and some processing
                frame_tensor = convert_tensor(frame_rgb_PIL) #img is 0/ 1 as soon as we convert it to a tensor from PIL image
                # Storing the tensor into the batch
                batch[i] = frame_tensor

                # im2 = ax2.imshow(frame_tensor[0],cmap='gray', interpolation='none')
                # im2.set_data(frame_tensor[0])
            
            # Processing the batch with network
            with torch.no_grad():
                handwritten_output = network(batch)

            # Printing/ visualizing the results on GUI
            for i in range(len(roi_array)):
                #create two subplots
                ax1 = plt.subplot(len(roi_array),2,2*i+1)
                ax1.set_title("Original image")
                plt.xticks([])
                plt.yticks([])

                prediction = handwritten_output.data.max(1, keepdim=True)[1][i].item()
                ax2 = plt.subplot(len(roi_array),2,2*i+2)
                ax2.set_title("Prediction: " + str(prediction))
                plt.xticks([])
                plt.yticks([])

                im1 = ax1.imshow(roi_array[i]) 
                im1.set_data(roi_array[i])

                im2 = ax2.imshow(batch[i][0],cmap='gray', interpolation='none') #Matplot lib expects input in RGB order but openCV is BGR
                im2.set_data(batch[i][0])

                cv2.putText(frame_color,str(prediction), (tl_array[i][0], tl_array[i][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                print("Prediction: {}".format(handwritten_output.data.max(1, keepdim=True)[1][i].item()))
                
            plt.pause(0.0001)

        cv2.imshow('Original Image', frame_color)

        # Useful info: If your tensor doesn’t contain a color channel dimension the colormap is undefined and matplotlib will use its default one. Src: https://discuss.pytorch.org/t/grayscale-image-plotted-to-have-colours/135860 
        
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    plt.close('all')
    cv2.destroyAllWindows()
    return(1)

if __name__ == "__main__":
    main()
