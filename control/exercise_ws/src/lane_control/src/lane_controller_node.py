#!/usr/bin/env python3
import numpy as np
import rospy
import debugpy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, SegmentList

from lane_controller.controller import PurePursuitLaneController

# 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1



class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.pp_controller = PurePursuitLaneController(self.params)

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)
        self.breakpoints_enabled=False

        self.sub_ground_projected_lanes = rospy.Subscriber("/agent/ground_projection_node/lineseglist_out",
                                                            SegmentList,
                                                            self.cbGroundProjectedLineSegments,
                                                            queue_size=1)

        self.log("Initialized!")
        debugpy.listen(5678)
        self.log("Waiting for debugger attach")

        self.right_offset = 0.12
        self.lookup_distance = 0.25
        self.lookup_depth = 0.05
        self.max_speed = 0.1
        self.K = 1
        self.last_omega=0
        self.last_v = self.max_speed
        #debugpy.wait_for_client()
        
        #self.log('break on this line')

    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        self.pose_msg = input_pose_msg

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        # TODO This needs to get changed
        car_control_msg.v = 0.5
        car_control_msg.omega = 0

        #if self.breakpoints_enabled:
        #    debugpy.breakpoint()
        #    self.log('break on this line')

        #self.publishCmd(car_control_msg)

    def cbGroundProjectedLineSegments(self, segments_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        self.segments_msg = segments_msg

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.segments_msg.header



        if self.breakpoints_enabled:
            debugpy.breakpoint()
            self.log('break on this line')

        yellow_lines = []
        yellow_lines_np = []
        white_lines = []
        white_lines_np = []

        for segment in segments_msg.segments:
            assert len(segment.points)==2
            # x is the distance from the front of the duckie.
            # y is the left-right distance. 
            start = (segment.points[0].x,segment.points[0].y)
            end = (segment.points[1].x,segment.points[1].y)
            if segment.color==1:
                #Yellow line.
                yellow_lines.append((start,end))
                yellow_lines_np.append([segment.points[0].x,
                                        segment.points[0].y,
                                        segment.points[1].x,
                                        segment.points[1].y])

            elif segment.color==0:
                #White line.
                white_lines.append((start,end))
                white_lines_np.append([segment.points[0].x,
                        segment.points[0].y,
                        segment.points[1].x,
                        segment.points[1].y])

        yellow_lines_np = np.array(yellow_lines_np)
        white_lines_np = np.array(white_lines_np)

        if self.breakpoints_enabled:
            debugpy.breakpoint()
            self.log('break on this line')

        lookup_distance =self.lookup_distance

        aim_y = 0
        aim_x = lookup_distance
        match=False

        car_control_msg.v = self.last_v
        car_control_msg.omega = self.last_omega

        #Basic line detection        
        if len(yellow_lines_np)>0:
            dist_p1 = np.sqrt(yellow_lines_np[:,0]**2 + yellow_lines_np[:,1]**2)
            #dist_p2 = np.sqrt(yellow_lines_np[:,2]**2 + yellow_lines_np[:,3]**2)
            valid_points = yellow_lines_np[(dist_p1 > 0.3) & (dist_p1 < 0.6)]
            if len(valid_points)>0:
                aim_x = valid_points[:,[0,2]].mean()
                aim_y = valid_points[:,[1,3]].mean() - self.right_offset
                match=True
                car_control_msg.v = self.max_speed
                alpha = np.arctan(aim_y/aim_x)
                car_control_msg.omega = np.sin(alpha) / self.K

        self.log(f"v={car_control_msg.v}, omega = {car_control_msg.omega:.2f}. Aiming at {aim_x:.2f}, {aim_y:.2f}. Match={match}")

        self.publishCmd(car_control_msg)

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)


    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
