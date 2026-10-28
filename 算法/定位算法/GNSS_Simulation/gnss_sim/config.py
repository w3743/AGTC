"""场景配置：田块、探针、土壤、大气、星座与定位模式定义。"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field, replace

# 2025-09-15 06:00:00 UTC（北京时间 14:00，白天电离层活跃时段）
T_START_UNIX = 1757916000.0


@dataclass(frozen=True)
class FieldConfig:
    """田块与参考点。默认取华北平原典型农田（河南）。"""
    size_m: float = 100.0
    lat0: float = 34.50
    lon0: float = 113.00
    h0: float = 80.0          # 大地高 (m)

    @property
    def half(self) -> float:
        return self.size_m / 2.0


@dataclass
class SessionConfig:
    """观测时段。"""
    t_start_unix: float = T_START_UNIX
    duration_s: float = 3600.0
    interval_s: float = 30.0

    @property
    def epochs(self) -> int:
        return int(self.duration_s // self.interval_s) + 1


@dataclass
class SoilConfig:
    """土壤与地表状态。moisture 为体积含水量 (m3/m3)。"""
    moisture: float = 0.20          # 0.05 干 / 0.20 适中 / 0.35 湿
    roughness_m: float = 0.02       # 地表均方根粗糙度
    canopy_height_m: float = 0.0    # 低矮作物层高度（0 = 裸土/出苗期）
    canopy_atten_db: float = 0.0    # 作物层附加衰减


@dataclass
class IonosphereConfig:
    """电离层状态。sfu 为 F10.7 太阳射电流量。"""
    sfu: float = 150.0              # 90 宁静 / 150 中等 / 220 活跃
    storm: bool = False             # 是否叠加磁暴扰动


@dataclass
class ConstellationConfig:
    systems: tuple = ("G", "C", "E")
    unhealthy_fraction: float = 0.03


@dataclass
class ProbeSpec:
    """单个探针的安装形态与天线类型。"""
    mount: str = "ground"           # ground 贴地安装 / mast 短杆架高
    antenna: str = "lowcost"        # lowcost 低成本微带 / geodetic 测量型扼流圈
    height_m: float | None = None   # 天线相位中心离地高度，None 按 mount 取值

    def resolved_height(self, rng=None) -> float:
        if self.height_m is not None:
            return self.height_m
        if self.mount == "ground":
            return 0.20 if rng is None else float(rng.uniform(0.12, 0.30))
        return 1.00 if rng is None else float(rng.uniform(0.60, 1.20))


@dataclass
class BaseStationConfig:
    """基准站配置（用于 DGNSS/RTK）。"""
    kind: str = "local"             # local 田块内基准站 / cors 15 km 外 CORS
    distance_km: float = 0.0        # local 时忽略；cors 时使用
    coordinate_source: str = "known"  # known 已知点 / spp 单点测量 / rtk 已标定
    spp_error_m: float = 2.0        # coordinate_source='spp' 时的坐标误差量级
    antenna: str = "geodetic"
    height_m: float = 2.00
    correction_age_s: float = 1.0   # 差分/RTK 改正数龄期


@dataclass
class ScenarioConfig:
    """完整场景配置。"""
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    session: SessionConfig = dc_field(default_factory=SessionConfig)
    soil: SoilConfig = dc_field(default_factory=SoilConfig)
    iono: IonosphereConfig = dc_field(default_factory=IonosphereConfig)
    const: ConstellationConfig = dc_field(default_factory=ConstellationConfig)
    probe: ProbeSpec = dc_field(default_factory=ProbeSpec)
    base: BaseStationConfig = dc_field(default_factory=BaseStationConfig)
    n_probes: int = 20
    min_spacing_m: float = 5.0
    seed: int = 20250915
    # 仰角掩蔽角（度）：贴地安装受地面遮挡，掩蔽角更高
    el_mask_ground: float = 12.0
    el_mask_mast: float = 10.0
    # 误差预算实验：需要关闭的误差源
    # 可选 "iono" / "tropo" / "multipath" / "ephemeris" / "noise" / "pcv"
    disable: tuple = ()

    def el_mask(self, mount: str) -> float:
        return self.el_mask_ground if mount == "ground" else self.el_mask_mast

    def copy_with(self, **kw) -> "ScenarioConfig":
        return replace(self, **kw)


@dataclass(frozen=True)
class ModeSpec:
    """定位模式定义。"""
    key: str
    label: str
    hardware: str
    correction: str
    note: str


MODES = {
    "spp_gps": ModeSpec(
        "spp_gps", "GPS 单点定位 (SPP)", "单频模块",
        "广播星历 + Klobuchar",
        "仅 GPS L1，最保守基线"),
    "spp_multi": ModeSpec(
        "spp_multi", "多系统单点定位 (SPP)", "单频模块",
        "广播星历 + Klobuchar + 多系统",
        "GPS+BDS+Galileo 联合解算"),
    "sbas": ModeSpec(
        "sbas", "星基增强 (BDSBAS/SBAS)", "单频模块",
        "SBAS 差分改正 + 电离层格网",
        "无需通信链路，覆盖中国及周边"),
    "dgnss": ModeSpec(
        "dgnss", "伪距差分 (DGNSS)", "单频模块",
        "本地基准站伪距改正",
        "基准站需电台/4G 链路"),
    "rtk": ModeSpec(
        "rtk", "载波相位差分 (RTK)", "双频 RTK 模块",
        "本地基准站双差 + 模糊度固定",
        "基准站距探针 <100 m"),
    "rtk_cors": ModeSpec(
        "rtk_cors", "网络 RTK (CORS)", "双频 RTK 模块",
        "15 km 外 CORS + NTRIP",
        "依赖 4G/NTRIP 服务"),
    "ppp": ModeSpec(
        "ppp", "精密单点定位 (PPP)", "双频 RTK 模块",
        "精密星历/钟差 + 非差相位",
        "无需基准站，需 20~30 min 收敛"),
    "ppp_ar": ModeSpec(
        "ppp_ar", "PPP-AR (模糊度固定)", "双频 RTK 模块",
        "精密产品 + 相位偏差改正",
        "收敛后精度接近 RTK"),
}

# 主实验使用的模式
MAIN_MODES = ("spp_gps", "spp_multi", "sbas", "dgnss", "rtk", "rtk_inst", "ppp")
