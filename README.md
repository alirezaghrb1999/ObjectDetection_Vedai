# Detecting Vehicle in Aerial Imagery ( vedai dataset )
in this project we aim to address vehicle detection (classification and localization) challenges by fine tuning three Deep Learning models : R-CNN , YOLOv5, RetiaNet
<br/>
The initial approach used in this project is the R-CNN model, which is a relatively simple model compared to others. However, it has limitations in terms of accuracy and implementation time. The selective search algorithm, responsible for object candidate box allocation, consumes more than 90% of the execution time. This issue is particularly problematic for real-time detection applications, making the use of this network less favorable.

To address these challenges, the problem is reframed as a classification problem. The project selects parts of the image that include the object of interest (e.g., a car) and parts that do not. Additionally, the model includes a background category. Figure bellow shows an example from the Vedai dataset where an image is classified as a vehicle during the R-CNN model's training process.
<br/>
![Capture](https://github.com/alirezaghrb1999/ObjectDetection_Vedai/assets/46087111/e2d5b6d4-1072-4a05-b7fb-f67d77cbcc95)
<br/>
After obtaining training data, a common convolutional network like VGG is used for classification. During testing, each image goes through the selective search algorithm to generate proposed boxes. These boxes are then fed into the model for prediction. If the model identifies a specific box as non-background, it is reported as containing an object.

Overall, this approach aims to improve accuracy and reduce implementation time by redefining the problem as a classification task and optimizing object candidate box allocation using selective search algorithm.
