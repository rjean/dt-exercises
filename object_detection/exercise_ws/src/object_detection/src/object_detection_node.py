#!/usr/bin/env python3
import numpy as np
import rospy
import rospkg
import debugpy
import os
import yaml
from cv_bridge import CvBridge
debugpy.listen(("localhost", 5678))

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point as PointMsg
from duckietown_msgs.msg import SegmentList, Segment, Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, AntiInstagramThresholds
from image_processing.anti_instagram import AntiInstagram
from image_processing.ground_projection_geometry import GroundProjectionGeometry, Point



import cv2
from object_detection.model import Wrapper
from cv_bridge import CvBridge

class ObjectDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )



        # Construct publishers
        self.pub_obj_dets = rospy.Publisher(
            "~duckie_detected",
            BoolStamped,
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION
        )

        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            "~image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )
        
        self.sub_thresholds = rospy.Subscriber(
            "~thresholds",
            AntiInstagramThresholds,
            self.thresholds_cb,
            queue_size=1
        )
        

        self.pub_seglist_filtered = rospy.Publisher("~seglist_filtered",
                                                    SegmentList,
                                                    queue_size=1,
                                                    dt_topic_type=TopicType.DEBUG)
                                                    
        self.pub_segmented_img = rospy.Publisher("~debug/segmented_image/compressed",
                                              CompressedImage,
                                              queue_size=1,
                                              dt_topic_type=TopicType.DEBUG)

        self.ai_thresholds_received = False
        self.anti_instagram_thresholds=dict()
        self.ai = AntiInstagram()
        self.bridge = CvBridge()

        model_file = rospy.get_param('~model_file','.')
        rospack = rospkg.RosPack()
        model_file_absolute = rospack.get_path('object_detection') + model_file
        self.model_wrapper = Wrapper(model_file_absolute)
        self.homography = self.load_extrinsics()
        homography = np.array(self.homography).reshape((3, 3))
        self.bridge = CvBridge()
        self.gpg = GroundProjectionGeometry(160,120, homography)
        self.initialized = True
        self.log("Initialized!")
    
    def thresholds_cb(self, thresh_msg):
        self.anti_instagram_thresholds["lower"] = thresh_msg.low
        self.anti_instagram_thresholds["higher"] = thresh_msg.high
        self.ai_thresholds_received = True

    def image_cb(self, image_msg):
        if not self.initialized:
            return

        # TODO to get better hz, you might want to only call your wrapper's predict function only once ever 4-5 images?
        # This way, you're not calling the model again for two practically identical images. Experiment to find a good number of skipped
        # images.

        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return
        
        # Perform color correction
        if self.ai_thresholds_received:
            image = self.ai.apply_color_balance(
                self.anti_instagram_thresholds["lower"],
                self.anti_instagram_thresholds["higher"],
                image
            )
        
        #image = cv2.resize(image, (224,224))
        img_small = cv2.resize(image, (160,120))
        self.model_wrapper.segment_cv2_image(img_small)
        yellow_segments_px = self.model_wrapper.get_yellow_segments_px()
        white_segments_px = self.model_wrapper.get_white_segments_px()

        yellow_segments = self.ground_project_segments_px(yellow_segments_px)
        #white_segments_px[0:60,0:160]=0 #Keep only white line of the left
        white_segments = self.ground_project_segments_px(white_segments_px)

        seg_msg = SegmentList()
        seg_msg.header = image_msg.header
        self.add_segments(yellow_segments, seg_msg, Segment.YELLOW)
        #self.add_segments(white_segments, seg_msg, Segment.WHITE)

        self.pub_seglist_filtered.publish(seg_msg)
        segmented_img_cv = cv2.applyColorMap(self.model_wrapper.seg*64, cv2.COLORMAP_JET)

        segmented_img = self.bridge.cv2_to_compressed_imgmsg(segmented_img_cv)
        segmented_img.header.stamp = image_msg.header.stamp
        self.pub_segmented_img.publish(segmented_img)

        print(f"Found {len(yellow_segments_px)} yellow segments")
            
        
        bboxes, classes, scores = self.model_wrapper.predict(image)

        
        msg = BoolStamped()
        msg.header = image_msg.header
        if len(bboxes)==0:
            #No detection at all!
            msg.data = None
        else:
            msg.data = self.det2bool(bboxes[0], classes[0]) # [0] because our batch size given to the wrapper is 1
        
        self.pub_obj_dets.publish(msg)

    def add_segments(self, yellow_segments, seg_msg, color):
        for yellow_segment in yellow_segments:
            new_segment = Segment()
            ground_pt_msg_1 = PointMsg()
            ground_pt_msg_1.z=0
            ground_pt_msg_1.x=yellow_segment[0][0]
            ground_pt_msg_1.y=yellow_segment[0][1]
            ground_pt_msg_2 = PointMsg()
            ground_pt_msg_2.z=0
            ground_pt_msg_2.x=yellow_segment[1][0]
            ground_pt_msg_2.y=yellow_segment[1][1]
            new_segment.points[0] = ground_pt_msg_1
            new_segment.points[1] = ground_pt_msg_2
            new_segment.color = color
            seg_msg.segments.append(new_segment)
    
        
    def ground_project_segments_px(self, segments_px):
        x=[]
        y=[]
        segments=[]
        for segment_px in segments_px:
            pixel1 = Point(segment_px[0][0]*4,segment_px[0][1]*4) #Conversion. Points are converted in 640x480 for the homography to work
            pixel2 = Point(segment_px[1][0]*4,segment_px[1][1]*4) #Conversion. Points are converted in 640x480 for the homography to work
            ground_projected_point1 = self.gpg.pixel2ground(pixel1)
            ground_projected_point2 = self.gpg.pixel2ground(pixel2)
            pt1 = (ground_projected_point1.x, ground_projected_point1.y)
            pt2 = (ground_projected_point2.x, ground_projected_point2.y)
            segment = (pt1,pt2)
            segments.append(segment)
        return segments

    def det2bool(self, bboxes, classes):
        # TODO remove these debugging prints
        print(bboxes)
        print(classes)
        
        # This is a dummy solution, remove this next line
        return len(bboxes) > 1
    
        
        # TODO filter the predictions: the environment here is a bit different versus the data collection environment, and your model might output a bit
        # of noise. For example, you might see a bunch of predictions with x1=223.4 and x2=224, which makes
        # no sense. You should remove these. 
        
        # TODO also filter detections which are outside of the road, or too far away from the bot. Only return True when there's a pedestrian (aka a duckie)
        # in front of the bot, which you know the bot will have to avoid. A good heuristic would be "if centroid of bounding box is in the center of the image, 
        # assume duckie is in the road" and "if bouding box's area is more than X pixels, assume duckie is close to us"
        
        
        obj_det_list = []
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            label = classes[i]
            
            # TODO if label isn't a duckie, skip
            # TODO if detection is a pedestrian in front of us:
            #   return True
    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.
        Returns:
            :obj:`numpy array`: the loaded homography matrix
        """
        # load intrinsic calibration
        cali_file_folder = '/data/config/calibrations/camera_extrinsic/'
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log("Can't find calibration file: %s.\n Using default calibration instead."
                     % cali_file, 'warn')
            cali_file = (cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(cali_file):
            msg = 'Found no calibration file ... aborting'
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        try:
            with open(cali_file,'r') as stream:
                calib_data = yaml.load(stream)
        except yaml.YAMLError:
            msg = 'Error in parsing calibration file %s ... aborting' % cali_file
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        return calib_data['homography']


if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name='object_detection_node')
    # Keep it spinning
    rospy.spin()
