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
from my_class import ThresholdTransform, MyNetwork
import os
from PIL import Image


def main():
    print("Hello World!")


    # Live video code based on Src: https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

    # main function code
    network = MyNetwork()
    network.load_state_dict(torch.load("./results/model_1A_1E.pth"))
    network.eval()

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
    # batch = torch.zeros(10, 1,28,28)#,dtype=float)

    # my_folder = "./handwritten_digits/"
    # i = 0
    # for filename in os.listdir(my_folder):
    #     img = Image.open(os.path.join(my_folder,filename))
    #     img = convert_tensor(img) #img is 0/ 1 here after binary
    #     # print("img is:" , img)
    #     batch[i] = img
    #     i = i+1

    # print("Batch shape: ", batch.shape)

    # # Get the NW output on handwritten digits
    # with torch.no_grad():
    #     handwritten_output = network(batch)

    # fig = plt.figure()
    # for i in range(10):
    #     plt.subplot(4,3,i+1)
    #     plt.tight_layout()
    #     plt.imshow(batch[i][0], cmap='gray', interpolation='none')
    #     plt.title("Prediction: {}".format(
    #         handwritten_output.data.max(1, keepdim=True)[1][i].item()))
    #     plt.xticks([])
    #     plt.yticks([])
    # fig
    # plt.show()



    # define a video capture object
    vid = cv2.VideoCapture(1)




    #Show live video using matplotlib: https://stackoverflow.com/questions/44598124/update-frame-in-matplotlib-with-live-camera-preview



        

    plt.ion()
    while(True):
        print(" ################################### LOOP STARTED ###################################")
        # Capture the video frame
        # by frame
        ret, frame_color = vid.read()

        
        frame_color = cv2.resize(frame_color, (500,500)) #cv::Mat
        #create two image plots
        frame_rgb = cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB) # cv::Mat
        

        frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
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
        pbibb = 20 #Pixel buffer in bb
        
        displayCnt = None
        roi_array = []
        # loop over the contours
        tl_array = []
        for c in cnts:
            # approximate the contour
            area = cv2.contourArea(c)
            if area > 300 and area < 3000:
                # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                cv2.drawContours(frame_gray, [c], -1, (0, 255, 0), 2)
                cv2.circle(frame_gray, (cX, cY), 7, (255, 255, 255), -1)
                # cv2.putText(frame_gray,str(area), (cX - 20, cY - 20),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                x,y,w,h = cv2.boundingRect(c)
                # draw the bounding rectangle | https://stackoverflow.com/questions/21104664/extract-all-bounding-boxes-using-opencv-python
                # print(frame_color.shape[0])
                tl_coord = (cX-w//2 - pbibb, cY-h//2 - pbibb)
                if (tl_coord[0] < pbibb or tl_coord[0] > frame_color.shape[0] - pbibb or tl_coord[1] < pbibb or tl_coord[1] > frame_color.shape[1] - pbibb) != 1:
                    # frame_gray = cv2.rectangle(frame_gray,(x-pbibb,y-pbibb),(x+max(w,h)+(pbibb//4),y+max(w,h)+(pbibb//4)),(0,0,255),1)
                    # roi = frame_color[y-pbibb:y+max(w,h)+(pbibb//4),x-pbibb:x+max(w,h)+(pbibb//4)]
                    new_width = w + 2*pbibb
                    new_height = h + 2*pbibb
                    # print(y-pbibb, " ", y+max(w,h)+pbibb, " ", x-pbibb, " ", x+max(w,h)+pbibb)
                    print(tl_coord[0], " ", new_width, " ", tl_coord[1], " ", new_height)

                    roi = frame_color[tl_coord[1]:tl_coord[1] + new_height,tl_coord[0]:tl_coord[0] + new_width]
                    frame_color = cv2.rectangle(frame_color,tl_coord,(tl_coord[0] + new_width,tl_coord[1] + new_height),(0,0,255),1)

                    # cv2.imshow('ROI', roi )
                    roi_array.append(roi)
                    tl_array.append(tl_coord)

                # cv2.waitKey(5000)


        
        # Notice the COLOR_BGR2RGB which means that the color is
        # converted from BGR to RGB

        
        

        # cv2.imshow('frame', frame)

        #Modes: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        # Src: https://www.geeksforgeeks.org/convert-opencv-image-to-pil-image-in-python/
        
        # im2 = ax2.imshow(frame_rgb_PIL)
        # im2.set_data(frame_rgb_PIL)
        

        

        print("Number of ROI: ", len(roi_array))
        # batch = torch.empty((1,1,28,28))
        if (len(roi_array) != 0):

            batch = torch.zeros(len(roi_array),1,28,28)#,dtype=float)
            
            for i in range(len(roi_array)):
                # cv2.imshow('ROI image: ' + str(i) , roi_array[i])
                # im3 = ax3.imshow(roi_array[i]) #Matplot lib expects input in RGB order but openCV is BGR
                # im3.set_data(roi_array[i])
                
                frame_rgb_PIL = Image.fromarray(roi_array[i], mode="RGB") #Convert the cv::Mat to PIL image which is fine
                frame_tensor = convert_tensor(frame_rgb_PIL) #img is 0/ 1 as soon as we convert it to a tensor from PIL image
                batch[i] = frame_tensor

                # im2 = ax2.imshow(frame_tensor[0],cmap='gray', interpolation='none')
                # im2.set_data(frame_tensor[0])
            
            with torch.no_grad():
                handwritten_output = network(batch)

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

                im1 = ax1.imshow(roi_array[i]) #Matplot lib expects input in RGB order but openCV is BGR
                im1.set_data(roi_array[i])

                im2 = ax2.imshow(batch[i][0],cmap='gray', interpolation='none') #Matplot lib expects input in RGB order but openCV is BGR
                im2.set_data(batch[i][0])

                cv2.putText(frame_color,str(prediction), (tl_array[i][0], tl_array[i][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                print("Prediction: {}".format(handwritten_output.data.max(1, keepdim=True)[1][i].item()))
                
            plt.pause(0.0001)

        cv2.imshow('Original Image', frame_color)


        # print("Tensor shape: ", frame_tensor.size())
        
        # If your tensor doesn’t contain a color channel dimension the colormap is undefined and matplotlib will use its default one. Src: https://discuss.pytorch.org/t/grayscale-image-plotted-to-have-colours/135860 
        
        # plt.imshow(frame)
        # plt.imshow(frame.permute(1, 2, 0))
        # plt.show()
        # print("img is:" , img)
        # plt.imshow(batch[0][0], cmap='gray', interpolation='none')

        # Get the NW output on live video
        


        
        # Put variable in matplotlib title: https://stackoverflow.com/questions/43757820/how-to-add-a-variable-to-python-plt-title
        # ax2.set_title('Detected digit: {}'.format(handwritten_output.data.max(1, keepdim=True)[1][0].item()))
        # print("Detected digit: ",handwritten_output.data.max(1, keepdim=True)[1][0].item() )
    
        # Displaying the Scanned Image by using cv2.imshow() method
        # Display the resulting frame

        # Displaying the converted image
        # pil_image.show()
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
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
