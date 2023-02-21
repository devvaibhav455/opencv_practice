Working alone for Project 2: Content Based Image Retrieval
Used one/ first time travel day.
LINK TO REPORT: https://wiki.khoury.northeastern.edu/display/~vaibhavd/Project+2%3A+Content-based+Image+Retrieval

I have no idea about the permission. Please let me know if you face any issue in accessing it.

Email: vaibhav.d@northeastern.edu

OS: Ubuntu 22.04
IDE: VS Code
OpenCV version: 4.5.4

I made two programs write_csv and read_csv to speed-up the CBIR process.
There is some code which is commented that I used for debugging and building up the project. Though, it can be removed to make the code look cleaner but I have not removed it for my future reference, to freshen up my basics.

############################ Instructions to execute binary ############################
NOTE: The csv file needs to be in the same directory as that of the binary.

usage:
 ./write_csv <directory path> <feature set> <distance_metric>(optional)
Valid feature sets:
	baseline
	hm :stands for Histogram Matching
	mhm :stands for Multi Histogram Matching
	tc: stands for Texture and Color
	cd: stands for Custom Design
	1: Extension 1: Gabor filters
	2: Extension 2: L5E5 rg chromaticity + rg chromaticity of original image
	3: Extension 3: Histogram matching for different distance metrics. Distance_metric is required for extension 3
Valid distance metric
	correlation/ chisquare/ intersection/ bhattacharyya

usage:
 ./read_csv <target image> <feature set><distance_metric>(optional)
Valid feature sets:
	baseline
	hm :stands for Histogram Matching
	mhm :stands for Multi Histogram Matching
	tc: stands for Texture and Color
	cd: stands for Custom Design
	1: Extension 1: Gabor filters
	2: Extension 2: L5E5 rg chromaticity + rg chromaticity of original image
	3: Extension 3: Histogram matching for different distance metrics. Distance_metric is required for extension 3
Valid distance metric
	correlation/ chisquare/ intersection/ bhattacharyya


Instructions for extensions:

EXTENSION 1) Gabor filter: 0, 90, 180, 270; 
	Example usage: write_csv ../images/ 1
EXTENSION 2) Collect all blue trash bins: write_csv 1
	Example usage: write_csv ../images/ 2
EXTENSION 3) Comparing 4 distance metrics. Distance metric is required as a compulsory argument to CLI for extension 3.: Correlation, Chi-Square, Histogram Intersection, and Bhattacharyya distance.
	Example usage: write_csv ../images/ 3 correlation