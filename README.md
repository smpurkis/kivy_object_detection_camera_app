# kivy object detection camera app
An example of object detection for mobile using Python. Specific example is with face detection. 

Aims:
- Show that object detection for mobile is possible using Python
- Make the code adaptable to be able to implement any object detection model

The packages used:
- [Kivy](https://kivy.org/#home), to create the graphical interface
- [Buildozer](https://buildozer.readthedocs.io/en/latest/), to package to mobile
- [Opencv](https://opencv.org/), to load the neural network and manipulate the images
- [Numpy](https://numpy.org/), also to manipulate the images
- [KivyMD](https://kivymd.readthedocs.io/en/latest/index.html), to make the graphics look good (files copied manually from the [repo](https://github.com/HeaTTheatR/KivyMD) due to this error I [encounted](https://www.reddit.com/r/kivy/comments/detase/buildozer_question_kivymd_importing_differently/))
- [XCamera](https://github.com/kivy-garden/xcamera), to get the camera feed (the default camera in Kivy doesn't work on Android for some reason)
- [Ultra fast face detection model](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB), used as an example to detect faces in real time

The main file is [kivy_object_detector.py](https://github.com/smpurkis/kivy_object_detection_camera_app/blob/master/kivy_object_detector.py).
With the basic structure being setup so that you can load your model in the `self.build` and add the prediction code to the `self.process_frame`.

Todo:
- Dig into XCamera, to see the method used to capture the camera feed when the default Kivy camera doesn't work
- Stay up to date with Opencv Python4Android recipes, currently only 4.0.21 can be used on Android using Buildozer
- restructure files
- add images to this README