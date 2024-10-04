# Object_segmentation_movement
This project implements an Object Segmentation Tool that utilizes deep learning techniques to segment and manipulate objects within images. Designed for ease of use, the tool allows users to specify an object of interest and generates a segmented image highlighting that object.

# Features
- Deep Learning-Based Segmentation: Utilizes the pre-trained DeepLabV3 model for state-of-the-art image segmentation.
- User Input for Object Segmentation: Allows users to define objects to be segmented without hardcoded class IDs.
- Object Relocation: After segmentation, users can move the segmented object within the image.
- Supports Popular Image Formats: Handles various image formats like .jpg and .png.
- Command-Line Interface: Simple command-line tool for ease of use.

# To run 
```
python obj.py --image "F:\\assignment_H3\\bagpack.jpg" --object "bagpack" --output "./generated.png"
```
