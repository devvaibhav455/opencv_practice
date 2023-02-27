Working alone for Project 3: Real-time 2-D Object Recognition 
Used two time travel days for this project. Total used: 1(Project 1); 2 (Project 2) = 3
LINK TO REPORT: https://wiki.khoury.northeastern.edu/display/~vaibhavd/Project+3%3A+Real-time+2-D+Object+Recognition

I have no idea about the permission. Please let me know if you face any issue in accessing it.

Email: vaibhav.d@northeastern.edu

OS: Ubuntu 22.04
IDE: VS Code
OpenCV version: 4.5.4

I made a program binProcess which has four mode of operations
There is some code which is commented that I used for debugging and building up the project. Though, it can be removed to make the code look cleaner but I have not removed it for my future reference, to freshen up my basics.

############################ Instructions to execute binary ############################
NOTE: The csv file needs to be in the same directory as that of the binary.

usage:
 ./binProcess <mode>
Valid modes:
	b: Basic training mode. Reads images from a directory and writes feature vectors to a csv file
	o: Object recognition mode. Takes image from user/ live feed and finds the closest match in the DB
	k: KNN classifier mode. Takes image from user and finds the closest match in the DB using KNN search
	t: Testing mode. Takes live feed from the camera to tune and test the system

'b' mode: 	./binProcess b
'o' mode: 	./binProcess o <image name>(optional)
			If image name is entered, an image for disk is read. Otherwise, camera live feed is used
'k' mode:	./binProcess <k>(int) <image name>
't' mode:	./binProcess t <image name>(optional)
			If image name is entered, an image for disk is read. Otherwise, camera live feed is used

Instructions for extensions:

EXTENSION 5) Comparing different distance metrics: Program is not made further complex by adding in a new CLI. Instead, a line of code is commented to test the new distance metric. Line #395 in modes.cpp 

No separate instructions for other extensions