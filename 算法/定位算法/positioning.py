import numpy as np
from scipy.optimize import least_squares

# 1. 基站相对坐标

stations = np.array([
    [0.0, 0.0],
    [8.0, 0.0],
    [2.0, 7.0],
    [8.0, 7.0],
    [4.0, 4.0],
    [5.0, 5.0]
])

# 2. 小车真实位置

car_real = np.array([4.5, 3.3])

# 3. 计算真实距离

def get_real_distances(position):
    return np.sqrt(
        (stations[:, 0] - position[0]) ** 2 +
        (stations[:, 1] - position[1]) ** 2
    )

# 4. 模拟测距

def get_measured_distances(real_distances):
    error = np.random.uniform(-0.10, 0.10, len(real_distances))
    return real_distances + error

# 5. 最小二乘法定位

def locate(measured_distances):

    # 计算某个假设位置的“理论距离”
    def residual(position):

        calculated_distances = np.sqrt(
            (stations[:, 0] - position[0]) ** 2 +
            (stations[:, 1] - position[1]) ** 2
        )

        # 理论距离 - 实际测量距离
        return calculated_distances - measured_distances

    # 初始猜测位置
    initial_position = np.array([5.0, 3.0])

    # 最小二乘优化
    result = least_squares(
        residual,
        initial_position
    )

    return result.x

# 6. 计算定位误差

def get_position_error(real_position, estimated_position):

    return np.linalg.norm(
        real_position - estimated_position
    )

# 7. 单次测试

real_distances = get_real_distances(car_real)

measured_distances = get_measured_distances(real_distances)

estimated_position = locate(measured_distances)

position_error = get_position_error(
    car_real,
    estimated_position
)


print("========== 基站坐标 ==========")

for i, station in enumerate(stations):
    print(
        f"基站{i + 1}: "
        f"({station[0]:.2f}, {station[1]:.2f})"
    )


print("\n========== 小车真实位置 ==========")
print(
    f"({car_real[0]:.2f}, {car_real[1]:.2f})"
)


print("\n========== 真实距离 ==========")

for i, distance in enumerate(real_distances):
    print(
        f"基站{i + 1} -> 小车: "
        f"{distance:.3f} m"
    )


print("\n========== 模拟测量距离 ==========")

for i, distance in enumerate(measured_distances):
    print(
        f"基站{i + 1} -> 小车: "
        f"{distance:.3f} m"
    )


print("\n========== 最小二乘法计算 ==========")

print(
    f"小车估计位置: "
    f"({estimated_position[0]:.2f}, "
    f"{estimated_position[1]:.2f})"
)


print("\n========== 定位结果 ==========")

print(
    f"定位误差: "
    f"{position_error * 100:.2f} cm"
)

# 8. 连续测试100次

errors = []

for i in range(100):

    # 每次重新模拟一次测距误差
    measured_distances = get_measured_distances(
        real_distances
    )

    # 根据测量距离重新定位
    estimated_position = locate(
        measured_distances
    )

    # 计算本次定位误差
    error = get_position_error(
        car_real,
        estimated_position
    )

    errors.append(error)


errors = np.array(errors)


print("\n")
print("================================")
print("        连续100次测试")
print("================================")

print(
    f"平均定位误差: "
    f"{np.mean(errors) * 100:.2f} cm"
)

print(
    f"最大定位误差: "
    f"{np.max(errors) * 100:.2f} cm"
)

print(
    f"最小定位误差: "
    f"{np.min(errors) * 100:.2f} cm"
)

print(
    f"定位误差标准差: "
    f"{np.std(errors) * 100:.2f} cm"
)