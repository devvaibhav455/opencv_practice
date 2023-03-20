Working alone for Project 4: Calibration and Augmented Reality
Used zero time travel days for this project. Total used: 1(Project 1); 2 (Project 2) = 3
LINK TO REPORT: https://wiki.khoury.northeastern.edu/display/~vaibhavd/Project+4%3A+Calibration+and+Augmented+Reality

I have no idea about the permission. Please let me know if you face any issue in accessing it.

Email: vaibhav.d@northeastern.edu

OS: Ubuntu 22.04
IDE: VS Code
OpenCV version: 4.5.4

I made two programs: calibAR and harrisCornerDetector
There is some code which is commented that I used for debugging and building up the project. Though, it can be removed to make the code look cleaner but I have not removed it for my future reference, to freshen up my basics.

############################ Instructions to execute binary ############################
NOTE: The "calibration_images" folder needs to be stored one directory level up with respect to the present working directory of the terminal.

usage:
 ./calibAR <mode>
Valid modes:
	c: Calculate camera calibration parameters
	p: Projection mode
	1: AR TV using Aruco tags
	2: Alter the target mode

'c' mode: 	./calibAR c
'p' mode: 	./calibAR p <image/video name>(optional)
			If image/ video name is entered, projection will be tried on that frame, otherwise on live video.
'1' mode:	./calibAR <1>(int) <image/video name to display>
'2' mode:	./calibAR t <image/video name>(optional)
			If image/video name is entered, projection and alteration will be tried on that frame, otherwise on live video.

Instructions for extensions:

1) Get your system working with multiple targets in the scene: No separate instructions for other extensions
2) Enable your system to use static images or pre-captured video sequences with targets and demonstrate inserting virtual objects into the scenes
	Please follow the instructions for mode 'p'
3) Alter target to make it not look like a target any more
	Please follow the instructions for mode '2'
4) Test out several different cameras and compare the calibrations and quality of the results
	Different images need to be put in "calibration_images" folder to compare different cameras
5) Augmented Reality Television
	Please follow the instructions for mode '1'
