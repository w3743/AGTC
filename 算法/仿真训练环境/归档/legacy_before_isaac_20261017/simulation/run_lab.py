#!/usr/bin/env python3
"""Run and verify the Gazebo RGB-D + UWB laboratory. All child processes owned."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from common import ROOT, RUNTIME, lock_run, output_dir

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=float, default=120)
parser.add_argument('--gui', action='store_true')
parser.add_argument('--manual', action='store_true', help='Do not send automatic velocity commands')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
lock = lock_run('mapping')
out = output_dir('mapping', args.output)
children=[]
handles=[]

def start(name, cmd):
    log = (out/(name+'.log')).open('w')
    handles.append(log)
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    children.append((name,process))
    return process

def stop():
    for _,p in reversed(children):
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGINT)
    deadline=time.monotonic()+12
    for _,p in children:
        try: p.wait(timeout=max(.1,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL)
            p.wait()
    for handle in handles: handle.close()

def execute():
    import numpy as np
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
    from sensor_msgs.msg import Image, CameraInfo
    from std_msgs.msg import Float64
    from nav_msgs.msg import Odometry
    from rtabmap_msgs.msg import MapData
    from cv_bridge import CvBridge
    import cv2

    start('gazebo',['gz','sim','-r','-v','3']+([] if args.gui else ['-s'])+[str(ROOT/'simulation/lab.sdf')])
    start('bridge',['ros2','run','ros_gz_bridge','parameter_bridge','--ros-args','-p',f'config_file:={ROOT}/simulation/bridge.yaml'])
    start('camera_tf',['ros2','run','tf2_ros','static_transform_publisher','--x','.26','--y','0','--z','.15','--roll','-1.57079632679','--pitch','0','--yaw','-1.57079632679','--frame-id','base_link','--child-frame-id','camera_optical_frame'])
    common=['--ros-args','-p','use_sim_time:=true','-p','frame_id:=base_link','-p','approx_sync:=false','-p','queue_size:=30','-p','qos:=1','-r','rgb/image:=/camera/color/image_raw','-r','depth/image:=/camera/depth/image_raw','-r','rgb/camera_info:=/camera/color/camera_info']
    start('visual_odometry',['ros2','run','rtabmap_odom','rgbd_odometry']+common+['-p','odom_frame_id:=visual_odom','-p','publish_tf:=true','-r','odom:=/visual/odom'])
    start('rtabmap',['ros2','run','rtabmap_slam','rtabmap']+common+['-p',f'database_path:={out}/map.db','-p','subscribe_depth:=true','-p','Grid/FromDepth:=true','-r','odom:=/visual/odom'])
    if args.gui:
        start('map_viewer',['ros2','run','rtabmap_viz','rtabmap_viz']+common+['-p','subscribe_depth:=true','-r','odom:=/visual/odom'])
    rclpy.init()
    node=rclpy.create_node('robotics_lab_evaluator')
    node.set_parameters([rclpy.parameter.Parameter('use_sim_time',value=True)])
    cmd=node.create_publisher(Twist,'/cmd_vel',10)
    estimate_pub=node.create_publisher(PoseWithCovarianceStamped,'/uwb/pose',10)
    counters=dict(rgb=0,depth=0,camera_info=0,wheel_odom=0,visual_odom=0,map_data=0)
    ranges={}; exact={}; estimates=[]; visual_positions=[]; wheel_positions=[]; map_nodes=set()
    cvbridge=CvBridge()
    subscriptions=[]
    anchor_xy=np.array(json.loads((ROOT/'simulation/anchors.json').read_text())['anchors_m'])[:,:2]

    def solve(values):
        r=np.array([values[i][0]/100 for i in range(1,5)])
        a=anchor_xy
        mat=2*(a[1:]-a[0])
        rhs=r[0]**2-r[1:]**2+(a[1:]**2).sum(axis=1)-(a[0]**2).sum()
        return np.linalg.lstsq(mat,rhs,rcond=None)[0]

    def on_image(msg,kind):
        counters[kind]+=1
        if counters[kind]==3:
            im=cvbridge.imgmsg_to_cv2(msg,desired_encoding='bgr8' if kind=='rgb' else 'passthrough')
            if kind=='depth':
                valid=np.isfinite(im)&(im>0)
                summary['depth_valid_fraction']=float(valid.mean())
                im=np.nan_to_num(im,nan=0,posinf=0,neginf=0)
                im=np.clip(im/10*255,0,255).astype('uint8')
            cv2.imwrite(str(out/(kind+'.png')),im)

    def on_info(msg): counters['camera_info']+=1
    def on_odom(msg,kind):
        counters[kind]+=1
        p=msg.pose.pose.position
        (visual_positions if kind=='visual_odom' else wheel_positions).append([p.x,p.y,p.z])
    def on_map(msg):
        counters['map_data']+=1
        map_nodes.update(msg.graph.poses_id)
    def on_range(msg,i,target): target[i]=(msg.data,time.monotonic())

    for topic,kind in [('/camera/color/image_raw','rgb'),('/camera/depth/image_raw','depth')]:
        subscriptions.append(node.create_subscription(Image,topic,lambda m,k=kind:on_image(m,k),qos_profile_sensor_data))
    subscriptions.append(node.create_subscription(CameraInfo,'/camera/color/camera_info',on_info,qos_profile_sensor_data))
    for topic,kind in [('/wheel/odom','wheel_odom'),('/visual/odom','visual_odom')]:
        subscriptions.append(node.create_subscription(Odometry,topic,lambda m,k=kind:on_odom(m,k),qos_profile_sensor_data))
    subscriptions.append(node.create_subscription(MapData,'/mapData',on_map,10))
    for i in range(1,5):
        for kind,target in [('distances',ranges),('distances_ground_truth',exact)]:
            subscriptions.append(node.create_subscription(Float64,f'/uwb_gz_simulator/{kind}/a{i}t1',lambda m,j=i,t=target:on_range(m,j,t),10))
    summary={'purpose':'environment acceptance, not trained-policy evaluation','range_unit_upstream':'centimeters','uwb_estimator':'2D multilateration from noisy ranges only; anchors have equal known height','ground_truth_usage':'evaluation only, reconstructed independently from exact ranges','started_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'automatic_motion':not args.manual}
    begin=time.monotonic(); last_estimate=0; last_print=0
    print('实验启动。窗口首次加载需要一些时间；Ctrl+C 可停止。',flush=True)
    try:
        while time.monotonic()-begin<args.seconds:
            rclpy.spin_once(node,timeout_sec=.05)
            elapsed=time.monotonic()-begin
            if not args.manual:
                move=Twist()
                if elapsed>20:
                    move.linear.x=.22; move.angular.z=.16
                cmd.publish(move)
            now=time.monotonic()
            if len(ranges)==4 and len(exact)==4 and now-last_estimate>.1 and all(now-v[1]<.6 for v in ranges.values()) and all(now-v[1]<.6 for v in exact.values()):
                last_estimate=now
                xy=solve(ranges); truth=solve(exact)
                estimates.append([elapsed,*xy.tolist(),*truth.tolist(),float(np.linalg.norm(xy-truth))])
                pose=PoseWithCovarianceStamped(); pose.header.stamp=node.get_clock().now().to_msg(); pose.header.frame_id='world'
                pose.pose.pose.position.x=float(xy[0]); pose.pose.pose.position.y=float(xy[1]); pose.pose.pose.orientation.w=1.
                # Uncalibrated placeholder uncertainty; do not use for fusion without calibration.
                pose.pose.covariance[0]=.25; pose.pose.covariance[7]=.25; pose.pose.covariance[14]=1e6
                pose.pose.covariance[21]=pose.pose.covariance[28]=pose.pose.covariance[35]=1e6
                estimate_pub.publish(pose)
            if elapsed-last_print>10:
                last_print=elapsed
                print(f'{elapsed:.0f}s RGB={counters["rgb"]} 视觉里程计={counters["visual_odom"]} 地图节点={len(map_nodes)} UWB定位样本={len(estimates)}',flush=True)
            dead=[(name,p.returncode) for name,p in children if p.poll() is not None]
            if dead: raise RuntimeError(f'子进程提前退出：{dead}；请查看 {out}')
    finally:
        cmd.publish(Twist())
        summary.update(counters=counters,map_nodes=len(map_nodes),uwb_samples=len(estimates),wall_seconds=time.monotonic()-begin)
        if estimates:
            data=np.array(estimates)
            np.savetxt(out/'uwb_positions.csv',data,delimiter=',',header='elapsed_s,estimate_x_m,estimate_y_m,truth_x_m,truth_y_m,error_m',comments='')
            summary['uwb_rmse_m']=float(np.sqrt(np.mean(data[:,-1]**2)))
            summary['uwb_truth_displacement_m']=float(np.linalg.norm(data[-1,3:5]-data[0,3:5]))
            summary['uwb_truth_max_excursion_m']=float(np.linalg.norm(data[:,3:5]-data[0,3:5],axis=1).max())
        if visual_positions:
            np.savetxt(out/'visual_positions.csv',visual_positions,delimiter=',',header='x_m,y_m,z_m',comments='')
            summary['visual_displacement_m']=float(np.linalg.norm(np.array(visual_positions[-1])-visual_positions[0]))
        # A closed patrol can end at its starting point; test excursion along the trajectory.
        summary['checks']={'rgb_depth':counters['rgb']>10 and counters['depth']>10,'visual_odometry':counters['visual_odom']>10,'visual_mapping':len(map_nodes)>=3,'uwb_positioning':len(estimates)>20,'rover_motion':summary.get('uwb_truth_max_excursion_m',0)>.3}
        (out/'acceptance.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
        node.destroy_node(); rclpy.shutdown()
        print('结果：',out,flush=True)
    return all(summary['checks'].values())

try:
    passed=execute()
except KeyboardInterrupt:
    passed=False
finally:
    stop()
raise SystemExit(0 if passed else 2)
