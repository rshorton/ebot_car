
import pathlib
import cv2

import depthai as dai

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

# From Donkey car
from tub_v2 import TubWriter
from datastore import TubHandler

use_nn = True
nn_sync = True

def get_model_path(model_name):
    return str(pathlib.Path(__file__).parent.absolute()) + '/models/' + model_name

class DonkeyCar(Node):
    def __init__(self):
        super().__init__('ebot_car')

        self.joy_msg = None
        self.cmd_vel_msg = None

        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 1)

        self.sub_joy = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            1)

        # Joy teleop is mapped to cmd_vel_raw.  This process receives the msg and either
        # forwards as is, or replaces with output of auto pilot (NN output)
        self.sub_cmd_vel = self.create_subscription(
            Twist,
            'cmd_vel_raw',
            self.cmd_vel_callback,
            1)

        # DonkeyCar Tub for saving training data
        inputs=['cam/image_array','user/angle', 'user/throttle']
        types=['image_array','float', 'float']
        tub_path = TubHandler(path="/home/elsabot/donkey_car_data").create_tub_path()
        self.tub_writer = TubWriter(tub_path, inputs=inputs, types=types)

        with dai.Device(self.create_pipeline()) as device:
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            if use_nn:
                qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            frame = None

            while True:
                rclpy.spin_once(self, timeout_sec=0.001)

                inDet = None
                if use_nn:
                    if nn_sync:
                        inRgb = qRgb.get()
                        inDet = qDet.get()
                    else:
                        inRgb = qRgb.tryGet()
                        inDet = qDet.tryGet()
                else:
                    inRgb = qRgb.tryGet()

                linear = None
                twist = None    

                if inDet is not None:
                    linear = inDet.getLayerFp16('StatefulPartitionedCall/model/n_outputs0/BiasAdd/Add')[0]
                    twist = inDet.getLayerFp16('StatefulPartitionedCall/model/n_outputs1/BiasAdd/Add')[0]    
                    self.get_logger().info("Model twist: %f,  linear: %f" % (twist, linear))

                if inRgb is not None:
                    frame = inRgb.getCvFrame()

                    # Left-top button on controller used to switch between NN control vs. manual control
                    if self.joy_msg == None or self.joy_msg.buttons[4] == 0:
                        if self.cmd_vel_msg != None:
                            if False:
                                if (self.cmd_vel_msg.linear.x > 0.05):
                                    self.cmd_vel_msg.linear.x = 0.05
                                if (self.cmd_vel_msg.angular.z > 0.2):
                                    self.cmd_vel_msg.angular.z = 0.2
                                elif (self.cmd_vel_msg.angular.z < -0.2):
                                    self.cmd_vel_msg.angular.z = -0.2

                            self.pub_cmd_vel.publish(self.cmd_vel_msg)
                            self.get_logger().info("                                            Joy twist: %f,  linear: %f" % (self.cmd_vel_msg.angular.z, self.cmd_vel_msg.linear.x))

                            # Save data to tub for training
                            self.tub_writer.run(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.cmd_vel_msg.linear.x, self.cmd_vel_msg.angular.z)
                            self.cmd_vel_msg = None

                    elif linear != None and twist != None:
                        msg = Twist()
                        # Fix - why do these need a scale adjustment?
                        msg.linear.x = linear*3
                        msg.angular.z = twist*6
                        self.pub_cmd_vel.publish(msg)

                if frame is not None:
                    frame = cv2.resize(frame, (int(640*1.4), int(360*1.4)), interpolation = cv2.INTER_AREA)
                    cv2.imshow("", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

            self.finish()
  
    def finish(self):
        self.get_logger().info("Stopping DonkeyCar tub writer")
        self.tub_writer.shutdown()

    def joy_callback(self, msg):
        self.joy_msg = msg

    def cmd_vel_callback(self, msg):
        self.cmd_vel_msg = msg

    def create_pipeline(self):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutRgb = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")

        # Properties
        camRgb.setPreviewSize(160, 120)
        camRgb.setInterleaved(False)
        camRgb.setFps(20)

        if use_nn:
            nn = pipeline.create(dai.node.NeuralNetwork)
            nnOut = pipeline.create(dai.node.XLinkOut)
            nnOut.setStreamName("nn")

            # Define a neural network that will make predictions based on the source frames
            nn.setBlobPath(get_model_path('donkey_car_2021_4_norm_inp_rev_4shave.blob'))            
            nn.setNumInferenceThreads(2)
            nn.input.setBlocking(False)

            # Linking
            if nn_sync:
                nn.passthrough.link(xoutRgb.input)
            else:
                camRgb.preview.link(xoutRgb.input)

            camRgb.preview.link(nn.input)
            nn.out.link(nnOut.input)
        else:
            camRgb.preview.link(xoutRgb.input)

        self.get_logger().debug("Pipeline created.")
        return pipeline

def main(args=None):
    rclpy.init(args=args)
    depthai_publisher = DonkeyCar()
    depthai_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


