"""Build a local, network-independent RGB-D rover and UWB experiment world."""
from pathlib import Path
import json
import random

OUT = Path(__file__).resolve().parents[1] / 'simulation'
OUT.mkdir(exist_ok=True)
parts = ['''<?xml version="1.0"?>
<sdf version="1.9"><world name="robotics_lab">
<physics name="default" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>1</real_time_factor></physics>
<gravity>0 0 -9.81</gravity>
<plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
<plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
<plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
<plugin filename="libUWBGazeboPlugin.so" name="custom::UWBGazeboSystem">
 <min_noise_stddev_cm>5</min_noise_stddev_cm><max_noise_stddev_cm>30</max_noise_stddev_cm>
 <min_dropout_probability>0.02</min_dropout_probability><max_dropout_probability>0.9</max_dropout_probability>
 <enable_nlos_dropout>true</enable_nlos_dropout><apply_pair_bias>false</apply_pair_bias>
</plugin>
<scene><ambient>0.7 0.7 0.7 1</ambient><background>0.12 0.15 0.2 1</background><shadows>false</shadows></scene>
<light name="sun" type="directional"><pose>0 0 8 0 0 0</pose><diffuse>0.9 0.9 0.9 1</diffuse><direction>-0.5 0.2 -1</direction></light>
<model name="ground_plane"><static>true</static><link name="ground"><collision name="collision"><geometry><plane><normal>0 0 1</normal><size>30 30</size></plane></geometry></collision><visual name="visual"><geometry><plane><normal>0 0 1</normal><size>30 30</size></plane></geometry><material><ambient>0.45 0.48 0.5 1</ambient><diffuse>0.45 0.48 0.5 1</diffuse></material></visual></link></model>
''']

def box(name, xyz, size, rgb, collision=True):
    geom = f'<geometry><box><size>{size}</size></box></geometry>'
    coll = f'<collision name="c">{geom}</collision>' if collision else ''
    parts.append(f'<model name="{name}"><static>true</static><pose>{xyz} 0 0 0</pose><link name="link">{coll}<visual name="v">{geom}<material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse></material></visual></link></model>')

for name, xyz, size in [('north','0 5 1.5','10 .15 3'),('south','0 -5 1.5','10 .15 3'),('east','5 0 1.5','.15 10 3'),('west','-5 0 1.5','.15 10 3')]:
    box('wall_'+name, xyz, size, '.8 .8 .8')
random.seed(42)
for side in range(4):
    for row in range(5):
        for col in range(16):
            shade = random.choice(['.08 .1 .12', '.9 .25 .1', '.15 .4 .85', '.85 .8 .2', '.9 .9 .9'])
            along = -4.6+col*.6
            z = .2+row*.5
            xyz = [f'{along} 4.91 {z}', f'{along} -4.91 {z}', f'4.91 {along} {z}', f'-4.91 {along} {z}'][side]
            box(f'marker_{side}_{row}_{col}', xyz, '.4 .02 .3' if side<2 else '.02 .4 .3', shade)
for i,(x,y) in enumerate([(-3,-2),(-3,1),(-1,3.8),(2,3.8),(3.8,-2)]):
    box(f'crate_{i}',f'{x} {y} .45','.55 .55 .9','.15 .45 .65')

anchors = [(-4,-4,1.2),(4,-4,1.2),(4,4,1.2),(-4,4,1.2)]
for i,(x,y,z) in enumerate(anchors,1):
    parts.append(f'<model name="r1_rover_anchor_{i}"><static>true</static><pose>{x} {y} {z} 0 0 0</pose><link name="uwb_anchor_{i}"><visual name="v"><geometry><sphere><radius>.1</radius></sphere></geometry><material><ambient>1 .6 .05 1</ambient><diffuse>1 .6 .05 1</diffuse></material></visual></link></model>')

# The upstream UWB plugin identifies tag carriers by the x500 prefix. This
# model is intentionally a ground rover, not a simulated x500 aircraft.
parts.append('''<model name="x500_lab_rover"><pose>0 0 .12 0 0 0</pose>
<link name="base_link"><pose>0 0 .12 0 0 0</pose><inertial><mass>4</mass><inertia><ixx>.08</ixx><iyy>.12</iyy><izz>.15</izz></inertia></inertial><collision name="body"><geometry><box><size>.5 .35 .15</size></box></geometry></collision><visual name="body"><geometry><box><size>.5 .35 .15</size></box></geometry><material><ambient>.1 .45 .9 1</ambient><diffuse>.1 .45 .9 1</diffuse></material></visual>
<sensor name="rgbd" type="rgbd_camera"><pose>.26 0 .15 0 0 0</pose><topic>/camera</topic><gz_frame_id>camera_optical_frame</gz_frame_id><always_on>true</always_on><update_rate>10</update_rate><camera><horizontal_fov>1.2</horizontal_fov><image><width>640</width><height>480</height><format>R8G8B8</format></image><clip><near>.05</near><far>12</far></clip><depth_camera><clip><near>.05</near><far>12</far></clip></depth_camera></camera></sensor></link>
<link name="uwb_tag_1"><pose>0 0 .43 0 0 0</pose><inertial><mass>.01</mass><inertia><ixx>.0001</ixx><iyy>.0001</iyy><izz>.0001</izz></inertia></inertial><visual name="v"><geometry><sphere><radius>.04</radius></sphere></geometry><material><ambient>1 .2 .1 1</ambient></material></visual></link>
<joint name="tag_joint" type="fixed"><parent>base_link</parent><child>uwb_tag_1</child></joint>
''')
for side,y in [('left',.23),('right',-.23)]:
    parts.append(f'''<link name="{side}_wheel"><pose>0 {y} 0 1.5707963 0 0</pose><inertial><mass>.3</mass><inertia><ixx>.002</ixx><iyy>.002</iyy><izz>.002</izz></inertia></inertial><collision name="c"><geometry><cylinder><radius>.12</radius><length>.05</length></cylinder></geometry><surface><friction><ode><mu>1</mu><mu2>1</mu2></ode></friction></surface></collision><visual name="v"><geometry><cylinder><radius>.12</radius><length>.05</length></cylinder></geometry><material><ambient>.08 .08 .08 1</ambient></material></visual></link>
<joint name="{side}_joint" type="revolute"><parent>base_link</parent><child>{side}_wheel</child><axis><xyz expressed_in="__model__">0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>''')
parts.append('''<link name="caster"><pose>-.19 0 -.065 0 0 0</pose><inertial><mass>.05</mass><inertia><ixx>.0001</ixx><iyy>.0001</iyy><izz>.0001</izz></inertia></inertial><collision name="c"><geometry><sphere><radius>.055</radius></sphere></geometry><surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision><visual name="v"><geometry><sphere><radius>.055</radius></sphere></geometry></visual></link><joint name="caster_joint" type="fixed"><parent>base_link</parent><child>caster</child></joint>
<plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive"><left_joint>left_joint</left_joint><right_joint>right_joint</right_joint><wheel_separation>.46</wheel_separation><wheel_radius>.12</wheel_radius><topic>/cmd_vel</topic><odom_topic>/wheel/odom</odom_topic><tf_topic>/wheel/tf</tf_topic><frame_id>wheel_odom</frame_id><child_frame_id>base_link</child_frame_id><odom_publish_frequency>30</odom_publish_frequency></plugin>
<plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher"><publish_link_pose>true</publish_link_pose><publish_sensor_pose>false</publish_sensor_pose><publish_collision_pose>false</publish_collision_pose><publish_visual_pose>false</publish_visual_pose><publish_nested_model_pose>false</publish_nested_model_pose><use_pose_vector_msg>true</use_pose_vector_msg><update_frequency>10</update_frequency></plugin>
</model></world></sdf>''')
(OUT/'lab.sdf').write_text('\n'.join(parts),encoding='utf-8')
(OUT/'anchors.json').write_text(json.dumps({'anchors_m':anchors,'tag_height_m':.55,'upstream_range_unit':'cm'},indent=2))
bridges=[]
def bridge(gz, ros, rt, gt, direction='GZ_TO_ROS'):
    bridges.append(dict(gz_topic_name=gz,ros_topic_name=ros,ros_type_name=rt,gz_type_name=gt,direction=direction))
bridge('/clock','/clock','rosgraph_msgs/msg/Clock','gz.msgs.Clock')
bridge('/cmd_vel','/cmd_vel','geometry_msgs/msg/Twist','gz.msgs.Twist','ROS_TO_GZ')
bridge('/wheel/odom','/wheel/odom','nav_msgs/msg/Odometry','gz.msgs.Odometry')
# Only visual odometry owns the base_link TF parent. Wheel odometry remains
# available as a reference topic and must not publish a competing TF parent.
bridge('/camera/image','/camera/color/image_raw','sensor_msgs/msg/Image','gz.msgs.Image')
bridge('/camera/depth_image','/camera/depth/image_raw','sensor_msgs/msg/Image','gz.msgs.Image')
bridge('/camera/camera_info','/camera/color/camera_info','sensor_msgs/msg/CameraInfo','gz.msgs.CameraInfo')
bridge('/model/x500_lab_rover/pose','/ground_truth/poses','tf2_msgs/msg/TFMessage','gz.msgs.Pose_V')
for i in range(1,5):
    for kind in ['distances','distances_ground_truth']:
        topic=f'/uwb_gz_simulator/{kind}/a{i}t1'
        bridge(topic,topic,'std_msgs/msg/Float64','gz.msgs.Double')
# JSON is a YAML subset and is accepted by ros_gz_bridge's yaml-cpp parser.
(OUT/'bridge.yaml').write_text(json.dumps(bridges,indent=2))
print('Generated', OUT)
