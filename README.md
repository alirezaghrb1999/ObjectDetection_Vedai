# Detecting Vehicles in Aerial Imagery ( vedai dataset )
in this project we aim to address vehicle detection (classification and localization) challenges by fine tuning three Deep Learning models : R-CNN , YOLOv5, RetiaNet
<br/>
The initial approach used in this project is the R-CNN model, which is a relatively simple model compared to others. However, it has limitations in terms of accuracy and implementation time. The selective search algorithm, responsible for object candidate box allocation, consumes more than 90% of the execution time. This issue is particularly problematic for real-time detection applications, making the use of this network less favorable.

To address these challenges, the problem is reframed as a classification problem. The project selects parts of the image that include the object of interest (e.g., a car) and parts that do not. Additionally, the model includes a background category. Figure bellow shows an example from the Vedai dataset where an image is classified as a vehicle during the R-CNN model's training process.
<br/>
<br/>
![Capture](https://github.com/alirezaghrb1999/ObjectDetection_Vedai/assets/46087111/e2d5b6d4-1072-4a05-b7fb-f67d77cbcc95)
<br/>
After obtaining training data, a common convolutional network like VGG is used for classification. During testing, each image goes through the selective search algorithm to generate proposed boxes. These boxes are then fed into the model for prediction. If the model identifies a specific box as non-background, it is reported as containing an object.

Overall, this approach aims to improve accuracy and reduce implementation time by redefining the problem as a classification task and optimizing object candidate box allocation using selective search algorithm.

<br/>
<br/>

In the next phase of the project, we explored the RetinaNet model, which is a fascinating and powerful network known for its unique hierarchical structure. This network strikes a great balance between accuracy and speed, performing on par with the Faster R-CNN in terms of accuracy while maintaining a speed comparable to real-time networks. For this project, we utilized the Retina model with transfer learning, and the evaluations pertaining to this model will be discussed in the following section of this chapter. To provide visual representation, a portion of the output from the RetinaNet component is displayed in the figure below. This figure showcases the output of the Retina model alongside its corresponding prediction labels for some of the test data.
<br/>
<br/>
![Capture](https://github.com/alirezaghrb1999/ObjectDetection_Vedai/assets/46087111/b4a4d152-e3d4-4d99-96b0-7cef2fce2e38)
<br/>
The image above illustrates the results of the Retina model on the test data. In this image, the green box represents true positives (TP), the yellow box represents false positives (FP), and the red box represents false negatives (FN).

<br/>
<br/>

