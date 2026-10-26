import sys
sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from newton_backend import _robot_builder
import newton, warp as wp

wp.init()
robot, wheel_joints, guide_joints, relations = _robot_builder()
print("robot joint_count? body_count?", len(robot.joint_type), len(robot.body_type))
main = newton.ModelBuilder(gravity=(0, 0, -9.81))
main.add_world(robot, label_prefix="w0_")
main.replicate(robot, world_count=1, label_prefixes=["w1_"])
model = main.finalize()
print("world_count", model.world_count, "bodies", model.body_count, "joints", model.joint_count,
      "coords", model.joint_coord_count, "dofs", model.joint_dof_count)
print("joint_q shape", model.joint_q.shape, "joint_qd shape", model.joint_qd.shape)
print("base coord", model.joint_q_start.numpy()[0], "base dof", model.joint_qd_start.numpy()[0])
print("joint_q_start head", model.joint_q_start.numpy()[:8])
print("joint_qd_start head", model.joint_qd_start.numpy()[:8])
print("joint_target_q_start", model.joint_target_q_start.numpy()[:8], "size", model.joint_target_q_start.shape)
