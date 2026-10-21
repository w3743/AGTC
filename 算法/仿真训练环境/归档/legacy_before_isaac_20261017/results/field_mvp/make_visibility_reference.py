"""Independent Python reference geometry for the field visibility UI checks."""
import json
import math
from pathlib import Path


def matmul(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(len(b)))
             for c in range(len(b[0]))] for r in range(len(a))]


def matvec(a, v):
    return [sum(row[k] * v[k] for k in range(len(v))) for row in a]


def transpose(a):
    return [list(col) for col in zip(*a)]


def camera_basis(pose, body_attitude):
    _x, _y, yaw, gy, gp = pose
    roll, pitch = body_attitude
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
    ry = [[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]]
    rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    body = matmul(matmul(rz, ry), rx)
    forward_local = [math.cos(gp) * math.cos(gy),
                     math.cos(gp) * math.sin(gy), math.sin(gp)]
    right_local = [math.sin(gy), -math.cos(gy), 0]
    forward = matvec(body, forward_local)
    right = matvec(body, right_local)
    # env._basis returns right x forward as camera-up.
    up = [right[1] * forward[2] - right[2] * forward[1],
          right[2] * forward[0] - right[0] * forward[2],
          right[0] * forward[1] - right[1] * forward[0]]
    return forward, right, up


def point(origin, pose, body_attitude, intrinsics, u, v, z):
    forward, right, up = camera_basis(pose, body_attitude)
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]
    nx, ny = (u - cx) / fx, (v - cy) / fy
    return [origin[i] + z * (forward[i] + nx * right[i] - ny * up[i])
            for i in range(3)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


CASES = [
    {
        "name": "yaw_zero_cardinal_flat",
        "pose": [10.0, 20.0, 0.0, 0.0, 0.0],
        "body_attitude": [0.0, 0.0],
        "origin": [10.0, 20.0, 0.4],
        "intrinsics": [[4.0, 0.0, 1.0], [0.0, 3.0, 0.5], [0.0, 0.0, 1.0]],
        "values": [[1.0, 1.5, 2.0], [2.5, 3.0, 4.0]],
        "valid": [[True, False, True], [True, True, True]],
    },
    {
        "name": "yaw_north_positive_body_pitch_negative_roll_negative_gimbal_pitch",
        "pose": [7.25, 11.5, math.pi / 2, 0.35, -0.22],
        "body_attitude": [-0.19, 0.28],  # [roll, pitch], matching env payload
        "origin": [7.25, 11.5, 0.92],
        "intrinsics": [[3.7, 0.0, 1.15], [0.0, 4.2, 0.43], [0.0, 0.0, 1.0]],
        "values": [[0.8, 1.2, 1.9], [2.4, 3.1, 3.7]],
        "valid": [[True, True, False], [True, False, True]],
    },
    {
        "name": "yaw_southwest_negative_body_pitch_positive_roll_positive_gimbal_pitch",
        "pose": [19.0, 3.0, -2.1, -0.6, 0.45],
        "body_attitude": [0.25, -0.31],
        "origin": [18.6, 2.8, 1.1],
        "intrinsics": [[5.0, 0.0, 0.8], [0.0, 4.0, 0.6], [0.0, 0.0, 1.0]],
        "values": [[1.1, 1.6, 2.2], [2.8, 3.3, 4.2]],
        "valid": [[False, True, True], [True, True, False]],
    },
]


def main():
    for case in CASES:
        forward, right, up = camera_basis(case["pose"], case["body_attitude"])
        points = []
        for v, row in enumerate(case["values"]):
            for u, z in enumerate(row):
                if case["valid"][v][u]:
                    p = point(case["origin"], case["pose"], case["body_attitude"],
                              case["intrinsics"], u, v, z)
                    ray = [p[i] - case["origin"][i] for i in range(3)]
                    points.append({"u": u, "v": v, "z": z, "world": p,
                                   "camera_z_dot": dot(ray, forward),
                                   "slant": math.sqrt(dot(ray, ray))})
        case["basis"] = {"forward": forward, "right": right, "up": up}
        case["points"] = points
        height, width = len(case["values"]), len(case["values"][0])
        case["frustum"] = [
            point(case["origin"], case["pose"], case["body_attitude"],
                  case["intrinsics"], u, v, 5.0)
            for u, v in [(-0.5, -0.5), (width - 0.5, -0.5),
                         (width - 0.5, height - 0.5), (-0.5, height - 0.5)]
        ]
    out = Path(__file__).with_name("visibility_reference.json")
    out.write_text(json.dumps(CASES, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
