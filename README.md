# ebot_car

ROS node implementing a Donkey Car control node

Example video:  https://youtu.be/5PBUIvnHb9w

The data collection facility (Tub Part) of Donkey car is used for storing the training data.  The collected data is uploaded during training step 1 below.

Training, see the Google Colab notebooks in the 'training' directory.
1. Donkey_Car_Training_using_Google_Colab.ipynb used to peform training.  Original notebook from https://colab.research.google.com/github/robocarstore/donkey-car-training-on-google-colab/blob/master/Donkey_Car_Training_using_Google_Colab.ipynb
2. h5_to_openvino_2021_4.ipynb is used to convert the .h5 format model created by step 1 to OpenVino format.
3. http://blobconverter.luxonis.com/ is used convert the OpenVino format model to a MyriadX blob for loading on OAK-1/D.

## Running

Set the venv for the Depthai python dependencies before running. Example:

````
pushd ~/depthai-python/; . venv/bin/activate; popd

    (Revise the above command to use your install of DepthAi.)

In first terminal:
[Run robot base driver node for your robot]

In second terminal:
ros2 launch ebot_car ebot_car.launch.py

In third terminal:
ros2 launch ebot_car joy_teleop.launch.py

````

Pressing the upper left button on the game controller (you will probably need to remap the button for your controller) engages the auto pilot mode.  When released, normal teleop mode is enabled.  Training data is recorded to a Donkey Car data Tub whenenver the A button is pressed.
