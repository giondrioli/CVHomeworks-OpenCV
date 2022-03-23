# CVHomeworks-OpenCV
Homeworks developed in C++ for the Computer Vision course.

- HW1 is related to the creation of the histograms of an image, the equalization of the image and the use of different kind of filters, such as the gaussian, the bilateral and the median filter.
To set the parameters, some trackbars are used for every filter. More information are provided in the report placed inside the folder.
In order to run the program (with Ubuntu for example) the library OpenCV is needed, as well as the g++ compiler.
Open a terminal and run the following two commands (one after the other):

  - `` g++ filter.cpp lab2.cpp -o lab2 `pkg-config --cflags --libs opencv4` ``

  - `` ./lab2 ``

This is not the only way to run the program, it depends on the specific implementation.

- HW2 is related to the creation of a panoramic image, starting from a set of images obtained by rotating the camera of a small amount for each image. In order to stitch them together, the SIFT 
and ORB algorithms have been utilized. More information are provided in the report placed inside the folder.
In order to run the program (with Ubuntu for example) the library OpenCV is needed, as well as the g++ compiler.
Open a terminal and run the following two commands (one after the other):

  - `` g++ lab4.cpp panoramic_utils2.cpp -o lab4 `pkg-config --cflags --libs opencv4` ``

  - `` ./lab4 ``

This is not the only way to run the program, it depends on the specific implementation.
