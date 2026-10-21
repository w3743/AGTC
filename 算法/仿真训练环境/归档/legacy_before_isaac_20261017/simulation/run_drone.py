#!/usr/bin/env python3
"""Local PX4 SITL flight smoke test; never connects to external vehicles."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from common import ROOT, RUNTIME, lock_run, output_dir

p=argparse.ArgumentParser()
p.add_argument('--gui',action='store_true')
p.add_argument('--takeoff',action='store_true',help='Arm simulated aircraft, take off, hover, land')
p.add_argument('--seconds',type=float,default=100)
args=p.parse_args()
lock=lock_run('drone')
root=ROOT
px4=RUNTIME/'src/PX4-Autopilot'
out=output_dir('drone')
env=os.environ.copy()
env['GZ_PARTITION']=env.get('GZ_PARTITION','robotics_lab')+'_px4'
env['GZ_DISTRO']='harmonic'
env['PATH']=str(RUNTIME/'venv/px4/bin')+':'+env['PATH']
if not args.gui: env['HEADLESS']='1'
log=(out/'px4.log').open('w')
process=subprocess.Popen(['make','px4_sitl','gz_x500'],cwd=px4,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
summary={'simulation_only':True,'px4_version':'v1.17.0','takeoff_requested':args.takeoff,'heartbeat':False,'max_relative_altitude_m':0.,'position_samples':0}
connection=None
begin=time.monotonic()
try:
    from pymavlink import mavutil
    connection=mavutil.mavlink_connection('udpin:127.0.0.1:14540',source_system=250)
    heartbeat=connection.wait_heartbeat(timeout=75)
    if heartbeat is None: raise RuntimeError('PX4 heartbeat timeout; see '+str(out/'px4.log'))
    summary['heartbeat']=True
    print('PX4 仿真已连接。此入口只使用本机 SITL；Ctrl+C 可结束。',flush=True)
    # PX4 console parameters are applied to this new isolated simulation only.
    process.stdin.write(b'param set COM_RC_IN_MODE 4\nparam set NAV_RCL_ACT 0\nparam set NAV_DLL_ACT 0\n')
    process.stdin.flush()
    stage='wait'; requested=0.; last_heartbeat=0.; last_print=0.
    while time.monotonic()-begin<args.seconds:
        elapsed=time.monotonic()-begin
        if process.poll() is not None: raise RuntimeError('PX4 exited early')
        if elapsed-last_heartbeat>1:
            connection.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,mavutil.mavlink.MAV_AUTOPILOT_INVALID,0,0,0)
            last_heartbeat=elapsed
        msg=connection.recv_match(type=['GLOBAL_POSITION_INT','LOCAL_POSITION_NED','COMMAND_ACK','HEARTBEAT'],blocking=True,timeout=.2)
        if msg is not None:
            if msg.get_type()=='GLOBAL_POSITION_INT':
                summary['position_samples']+=1
                summary['max_relative_altitude_m']=max(summary['max_relative_altitude_m'],msg.relative_alt/1000.)
            if msg.get_type()=='COMMAND_ACK':
                summary.setdefault('command_acks',[]).append({'command':msg.command,'result':msg.result})
            if msg.get_type()=='HEARTBEAT':
                summary['last_armed']=bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        if args.takeoff and elapsed>25 and stage=='wait':
            connection.mav.command_long_send(connection.target_system,connection.target_component,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,0,1,0,0,0,0,0,0)
            stage='arming'; requested=elapsed
        if stage=='arming' and summary.get('last_armed'):
            connection.mav.command_long_send(connection.target_system,connection.target_component,mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,0,0,0,0,float('nan'),float('nan'),float('nan'),3.)
            stage='flying'; requested=elapsed
        if stage=='flying' and elapsed-requested>25:
            connection.mav.command_long_send(connection.target_system,connection.target_component,mavutil.mavlink.MAV_CMD_NAV_LAND,0,0,0,0,float('nan'),float('nan'),float('nan'),0)
            stage='landing'
        if elapsed-last_print>10:
            last_print=elapsed
            print(f'{elapsed:.0f}s 阶段={stage} 最大相对高度={summary["max_relative_altitude_m"]:.2f}m',flush=True)
except KeyboardInterrupt:
    summary['user_stopped']=True
except Exception as exc:
    summary['error']=str(exc)
    print(exc,flush=True)
finally:
    if connection: connection.close()
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGINT)
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL); process.wait()
    log.close()
    summary['flight_observed']=summary['max_relative_altitude_m']>1.
    summary['wall_seconds']=time.monotonic()-begin
    (out/'acceptance.json').write_text(json.dumps(summary,indent=2))
    print('无人机记录：',out,flush=True)
raise SystemExit(0 if summary['heartbeat'] and (not args.takeoff or summary['flight_observed']) and 'error' not in summary else 2)
