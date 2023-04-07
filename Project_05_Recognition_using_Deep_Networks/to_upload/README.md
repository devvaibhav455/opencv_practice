Working alone for Project 5: Recognition using Deep Networks
Used 3 time travel days for this project. Total used: 1(Project 1); 2 (Project 2); 3(Project 5) = 6
LINK TO REPORT: https://wiki.khoury.northeastern.edu/display/~vaibhavd/Project+5%3A+Recognition+using+Deep+Networks

Google drive link for the images used to train: https://drive.google.com/drive/folders/14N7GMC9Hl3LRxTG1YTCOV34vCJFPut_Z?usp=sharing

I have no idea about the permission. Please let me know if you face any issue in accessing it.

Email: vaibhav.d@northeastern.edu

OS: Ubuntu 22.04
IDE: VS Code
OpenCV version: 4.5.4

I made a helper file which contains the class definition of the network and train/ test function, function to process image: 
There is some code which is commented that I used for debugging and building up the project. Though, it can be removed to make the code look cleaner but I have not removed it for my future reference, to freshen up my basics.

############################ Instructions to execute binary ############################
NOTE: The "calibration_images" folder needs to be stored one directory level up with respect to the present working directory of the terminal.

Six python scripts are made.

mnist_model_1A_1E.py     --> 	Tasks 1A to 1E
mnist_model_1F_1G.py     --> 	Tasks 1F to 1G
mnist_model_2.py	     --> 	Task 2
mnist_model_3_greek.py	 --> 	Task 3
mnist_model_4.py	     --> 	Task 4
mnist_model_ext1.py	     --> 	Extension 1 : Live video digit recognition

usage:
python3 script_name <mode>(optional)
Valid modes for mnist_model_3_greek.py only:
	1: Testing on handwritten greek letters: alpha, beta and gamma
	2: Testing on extra handwritten greek letters: theta, omega and pi

Instructions for extensions:

1) Live video digit recognition application using the trained network: No separate instruction other than running the script
2) Evaluate more dimensions on task 4: Already done. The script also logs the results to a CSV file so that it is easy to do the analysis.
3) Try more greek letters than alpha, beta, and gamma
	Please follow the instructions for mode '2'
