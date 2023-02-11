Working alone for Project 2: Content Based Image Retrieval
LINK TO REPORT: https://wiki.khoury.northeastern.edu/display/~vaibhavd/Project+2%3A+Content-based+Image+Retrieval

I have no idea about the permission. Please let me know if you face any issue in accessing it.

Email: vaibhav.d@northeastern.edu

OS: Ubuntu 22.04
IDE: VS Code
OpenCV version: 4.5.4

I made two programs write_csv and read_csv to speed-up the CBIR process.

Instructions to execute binary

usage:
 ./write_csv <directory path> <feature set>
Valid feature sets:
	baseline
	hm :stands for Histogram Matching
	mhm :stands for Multi Histogram Matching
	tc: stands for Texture and Color
	cd: stands for Custom Design
	1: Extension 1 | Gabor filters
	2: Extension 2: L5E5 rg chromaticity + rg chromaticity of original image
    3: Extension 3: Comparing 4 histogram distance metrics


usage:
 ./read_csv <target image> <feature set>
Valid feature sets:
	baseline
	hm :stands for Histogram Matching
	mhm :stands for Multi Histogram Matching
	tc: stands for Texture and Color
	cd: stands for Custom Design
	1: Extension 1 | Gabor filters
	2: Extension 2: L5E5 rg chromaticity + rg chromaticity of original image
    3: Extension 3: Comparing 4 histogram distance metrics


Instructions for extensions:

EXTENSION 1)  Gabor filter
EXTENSION 2) Collect all blue trash bins:
EXTENSION 3) Comparing 4 distance metrics: Correlation, Chi-Square, Histogram Intersection, and Bhattacharyya distance.