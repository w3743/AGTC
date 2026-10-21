"""Read-only, single-environment instrumentation of the viewer's SB3 policy.

Actor tensors come from the actual action call. The optional critic is a
separate diagnostic forward with its own recurrent history. No training code,
parameters, observations, or actor recurrent states are replaced.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution

from ..action_contract import ACTION_LABELS, action_config

class PolicyInspector:
    def __init__(self, runner):
        self.runner = runner
        self.policy = runner.model.policy
        self.modules = dict(self.policy.named_modules())
        self.layers: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.sequence = 0
        self.snapshot: dict[str, Any] | None = None
        self.critic_state = None
        self._phase = "actor"
        self._errors: list[str] = []
        self._branch_dims: dict[str, int] = {}
        self._branch_terminal_keys: dict[str, str] = {}
        self._make_graph()

    def _node(self, key, title, description, branch, column, parents=(), *, module=None,
              summary=False, source="实际前向"):
        self.layers[key] = dict(key=key, title=title, description=description,
                                branch=branch, column=float(column), parents=tuple(parents),
                                module=module, summary=summary, source=source, values=None)

    def _make_graph(self):
        action_mean_column = 13
        self._branch_dims = self._resolve_branch_dims()
        towers = (("lidar", "laser", "激光雷达", "距离/8米与有效掩码，2×400"),
                  ("depth", "depth", "深度图", "轴向Z深度/量程，1×160×88；无效值为0且无单独掩码；不是RGB图"),
                  ("detections", "boxes", "检测框", "16个候选×10字段；无效行由presence掩码排除"),
                  ("state", "proprioception", "任务与定位状态", "39维任务/定位状态；不含车体速度和横滚俯仰真值"),
                  ("feedback", "feedback", "惯性与驱动反馈", "41维：云台IMU、实测关节、左右电机Iq/RPM及有效性/帧龄；仿真电流为显式驱动力矩模型估计"))
        terminals = []
        for branch, attribute, title, description in towers:
            previous = f"input.{branch}"
            self._node(previous, title + "输入", description, branch, 0)
            tower = getattr(self.policy.features_extractor, attribute)
            leaves = [(name, module) for name, module in tower.named_modules() if name and not list(module.children())]
            for index, (name, module) in enumerate(leaves, 1):
                key = f"features_extractor.{attribute}.{name}"
                self._node(key, self._module_title(module), self._module_description(module), branch, index,
                           (previous,), module=key, summary=index == len(leaves))
                previous = key
            self._branch_terminal_keys[branch] = previous
            branch_dim = self._branch_dims.get(branch)
            self.layers[previous]["title"] = title + "编码输出" + (f" · {branch_dim}维" if branch_dim else "")
            if branch == "detections":
                self.layers[previous]["summary"] = False
                self._node("detections.masked", "逐框有效性掩码",
                           f"逐框{branch_dim}维特征 × 输入第8列presence；按真实算式重建" if branch_dim else
                           "逐框特征 × 输入第8列presence；按真实算式重建",
                           branch, len(leaves) + 1, (previous, "input.detections"), source="算式重建")
                self._node("detections.pooled", "检测编码输出" + (f" · {branch_dim}维" if branch_dim else ""),
                           "沿16个候选逐特征amax；不是求平均", branch,
                           len(leaves) + 2, ("detections.masked",), summary=True, source="算式重建")
                self._node("detections.winner", "池化来源索引", "每个输出特征来自哪个候选行；并列最大值取首行，空槽可能在零值并列中出现",
                           branch, len(leaves) + 3, ("detections.masked",), source="算式重建")
                previous = "detections.pooled"
            terminals.append(previous)
        fusion_dim = sum(self._branch_dims.values()) if len(self._branch_dims) == len(towers) else None
        self._node("features_extractor", "五路特征拼接" + (f" · {fusion_dim}维" if fusion_dim else ""),
                   self._fusion_description(fusion_dim),
                   "fusion", 9, terminals, module="features_extractor", summary=True)
        for role, branch, path in (("actor", "actor", "policy_net"), ("critic", "critic", "value_net")):
            lstm_key = f"lstm_{role}"
            lstm = getattr(self.policy, lstm_key, None)
            source = "实际驾驶前向" if role == "actor" else "价值旁路诊断"
            if isinstance(lstm, nn.LSTM):
                self._node(lstm_key, "驾驶记忆 LSTM" if role == "actor" else "价值记忆 LSTM",
                           str(lstm) + "；h是短时输出，c是内部记忆；价值支路不控制车辆", branch, 14,
                           (lstm_key + ".h",), module=lstm_key, summary=role == "actor", source=source)
                for i, (suffix, label) in enumerate((("h0", "输入隐状态 h₀"), ("c0", "输入记忆 c₀"),
                                                     ("h", "输出隐状态 h₁"), ("c", "输出记忆 c₁"),
                                                     ("i", "输入门 i"), ("f", "遗忘门 f"),
                                                     ("g", "候选记忆 g"), ("o", "输出门 o"))):
                    reconstructed = suffix in ("i", "f", "g", "o")
                    if suffix in ("h0", "c0"):
                        description = ("上一决策步输出的循环状态，作为本次LSTM输入；跨步反馈折叠在此节点，不画跨帧回边；"
                                       "回合开始时清零" if role == "actor" else
                                       "上一价值诊断步输出的循环状态，作为本次LSTM输入；跨步反馈折叠在此节点，不画跨帧回边；"
                                       "诊断开始或回合开始时清零")
                        parents = ()
                        column = 8
                    elif suffix in ("i", "f", "g", "o"):
                        description = ("使用LSTM输入权重、循环权重及偏置，根据features_extractor与h₀重建；"
                                       "并以真实h₁/c₁核验")
                        parents = ("features_extractor", lstm_key + ".h0")
                        column = 11
                    elif suffix == "c":
                        description = "c₁ = f·c₀ + i·g；由本次门值及上一时刻记忆状态计算"
                        parents = tuple(lstm_key + "." + gate for gate in ("f", "i", "g")) + (lstm_key + ".c0",)
                        column = 12
                    else:  # h
                        description = "h₁ = o·tanh(c₁)；本次隐状态输出，供下一决策步使用"
                        parents = (lstm_key + ".o", lstm_key + ".c")
                        column = 13
                    self._node(f"{lstm_key}.{suffix}", label,
                               description, f"{role}_memory", column, parents, module=lstm_key,
                               source=("权重重建并核验" if reconstructed else source))
                previous = lstm_key
            else:
                # Canonical field checkpoints have independent actor/critic LSTMs.
                previous = "features_extractor"
            net = getattr(self.policy.mlp_extractor, path)
            leaves = [(name, module) for name, module in net.named_modules() if name and not list(module.children())]
            network_start = 15 if isinstance(lstm, nn.LSTM) else 11
            for index, (name, module) in enumerate(leaves, network_start):
                key = f"mlp_extractor.{path}.{name}"
                self._node(key, ("动作头 " if role == "actor" else "价值头 ") + self._module_title(module),
                           str(module), branch, index, (previous,), module=key,
                           summary=False, source=source)
                previous = key
            head = "action_net" if role == "actor" else "value_net"
            self._node(head, "动作分布均值 μ" if role == "actor" else "状态价值 V(s)",
                       str(getattr(self.policy, head)) + ("；均值尚未经过动作范围裁剪" if role == "actor" else "；额外只读计算，不参与驾驶"),
                       branch, network_start + len(leaves), (previous,), module=head, summary=False, source=source)
            if role == "actor":
                action_mean_column = network_start + len(leaves)
        action_dist = getattr(self.policy, "action_dist", None)
        log_std = getattr(self.policy, "log_std", None)
        use_sde = bool(getattr(self.policy, "use_sde", False)) or isinstance(action_dist, StateDependentNoiseDistribution)
        if isinstance(log_std, torch.Tensor):
            if use_sde:
                log_std_title = "gSDE 原始 log_std 参数"
                log_std_description = ("实际策略的可训练gSDE参数矩阵；有效噪声尺度依赖actor状态与探索矩阵。"
                                       "当前拓扑仅展开对角Gaussian σ，不绘制状态相关σ节点")
            else:
                log_std_title = "Gaussian log_std 参数"
                log_std_description = "实际策略的全局可训练参数；独立于动作均值 μ，σ = exp(log_std)"
            self._node("action.log_std", log_std_title, log_std_description,
                       "actor", action_mean_column + 1, (), source="实际分布参数", summary=False)
            if isinstance(action_dist, DiagGaussianDistribution) and not use_sde:
                self._node("action.std", "动作分布标准差 σ",
                           "实际对角Gaussian分布的stddev，等于exp(log_std)；它独立于μ，确定性驾驶不采样噪声",
                           "actor", action_mean_column + 2, ("action.log_std",), source="实际分布", summary=False)
        self._node(
            "action.applied",
            "四维输出动作",
            "动作分量依次为" + "、".join(ACTION_LABELS)
            + "；确定性均值经 SB3 裁剪 / 反归一化后范围为 [-1,1]。"
            "前两维分别乘最大履带速度，形成左右独立履带速度指令；"
            "后两维保持云台偏航 / 俯仰角速度语义。",
            "output", action_mean_column + 2, ("action_net",), summary=True,
        )

    @staticmethod
    def _module_title(module):
        if isinstance(module, nn.Linear):
            return f"全连接 {module.in_features}→{module.out_features}"
        if isinstance(module, nn.Conv1d):
            title = f"一维卷积 {module.in_channels}→{module.out_channels}"
            return title + (" · 环形填充" if module.padding_mode == "circular" else "")
        if isinstance(module, nn.Conv2d):
            return f"二维卷积 {module.in_channels}→{module.out_channels}"
        if isinstance(module, nn.AdaptiveAvgPool1d):
            return f"一维自适应平均池化 · {PolicyInspector._format_pool_size(module.output_size)}"
        if isinstance(module, nn.AdaptiveAvgPool2d):
            return f"二维自适应平均池化 · {PolicyInspector._format_pool_size(module.output_size)}"
        if isinstance(module, nn.Flatten):
            return "展平 Flatten"
        return type(module).__name__

    @staticmethod
    def _format_pool_size(size):
        if isinstance(size, int):
            return str(size)
        if isinstance(size, (tuple, list)):
            return "×".join(str(int(value)) for value in size)
        return str(size)

    @staticmethod
    def _module_description(module):
        description = str(module)
        if isinstance(module, nn.Conv1d) and module.padding_mode == "circular":
            description += "；环形填充把雷达首尾视为相邻方向"
        return description

    def _resolve_branch_dims(self) -> dict[str, int]:
        """Read declared output widths, falling back to each branch's terminal layer."""
        extractor = self.policy.features_extractor
        declared = getattr(extractor, "branch_dims", None)
        aliases = {
            "lidar": ("lidar", "laser"),
            "depth": ("depth",),
            "detections": ("detections", "boxes"),
            "state": ("state", "proprioception"),
            "feedback": ("feedback",),
        }
        attributes = {"lidar": "laser", "depth": "depth", "detections": "boxes", "state": "proprioception", "feedback": "feedback"}
        dims: dict[str, int] = {}
        for branch, names in aliases.items():
            if isinstance(declared, Mapping):
                for name in names:
                    value = declared.get(name)
                    if isinstance(value, (int, np.integer)) and int(value) > 0:
                        dims[branch] = int(value)
                        break
            if branch in dims:
                continue
            tower = getattr(extractor, attributes[branch], None)
            leaves = [module for name, module in tower.named_modules() if name and not list(module.children())] if isinstance(tower, nn.Module) else []
            for module in reversed(leaves):
                if isinstance(module, nn.Linear):
                    dims[branch] = int(module.out_features)
                    break
                if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    dims[branch] = int(module.out_channels)
                    break
        return dims

    def _fusion_description(self, fusion_dim: int | None) -> str:
        labels = (("lidar", "雷达"), ("depth", "深度"), ("detections", "检测"), ("state", "状态"), ("feedback", "反馈"))
        widths = [f"{label}{self._branch_dims[branch]}" for branch, label in labels if branch in self._branch_dims]
        if len(widths) == len(labels) and fusion_dim:
            return f"按{'、'.join(widths)}拼成{fusion_dim}维；没有注意力权重"
        return "按五路编码输出拼接；没有注意力权重"

    def _refresh_runtime_dimensions(self) -> None:
        """Use captured branch and fusion tensors to keep labels tied to the live forward."""
        for branch, key in self._branch_terminal_keys.items():
            value = self.layers.get(key, {}).get("values")
            if branch == "detections":
                value = self.layers.get("detections.pooled", {}).get("values")
            if value is not None and np.asarray(value).size:
                self._branch_dims[branch] = int(np.asarray(value).shape[-1])
                title = {"lidar": "激光雷达", "depth": "深度图", "detections": "检测框", "state": "任务与定位状态", "feedback": "惯性与驱动反馈"}[branch]
                self.layers[key]["title"] = title + f"编码输出 · {self._branch_dims[branch]}维"
                if branch == "detections":
                    self.layers["detections.pooled"]["title"] = f"检测编码输出 · {self._branch_dims[branch]}维"
                    self.layers["detections.masked"]["description"] = (
                        f"逐框{self._branch_dims[branch]}维特征 × 输入第8列presence；按真实算式重建"
                    )
        fusion = self.layers.get("features_extractor", {}).get("values")
        if fusion is not None and np.asarray(fusion).size:
            fusion_dim = int(np.asarray(fusion).shape[-1])
            self.layers["features_extractor"]["title"] = f"五路特征拼接 · {fusion_dim}维"
            self.layers["features_extractor"]["description"] = self._fusion_description(fusion_dim)

    def reset(self):
        self.critic_state = None
        self.snapshot = None
        for layer in self.layers.values():
            layer["values"] = None

    def parameters(self, key: str) -> dict[str, np.ndarray]:
        if key == "action.log_std":
            return {"log_std": self.policy.log_std.detach().cpu().numpy().copy()}
        path = self.layers[key].get("module")
        module = self.modules.get(path)
        if module is None:
            return {}
        return {name: value.detach().cpu().numpy().copy() for name, value in module.named_parameters(recurse=False)}

    def _save(self, key: str, tensor, *, strip_batch=True):
        if key not in self.layers or not isinstance(tensor, torch.Tensor):
            return
        value = tensor.detach().cpu().numpy()
        if strip_batch and value.ndim and value.shape[0] == 1:
            value = value[0]
        self.layers[key]["values"] = np.array(value, copy=True)

    def _lstm(self, key, module, inputs, output):
        sequence, (h, c) = output
        self._save(key, sequence[:, 0], strip_batch=True)
        h0, c0 = inputs[1]
        for suffix, value in (("h0", h0), ("c0", c0), ("h", h), ("c", c)):
            self._save(key + "." + suffix, value[:, 0], strip_batch=False)
        if key == "lstm_critic":
            self.critic_state = (h.detach().clone(), c.detach().clone())
        # nn.LSTM exposes h/c publicly, not gate tensors. Reconstruct only the
        # one-step, unidirectional, unprojected evaluation case and verify it.
        x = inputs[0]
        if x.shape[0] != 1 or module.bidirectional or module.proj_size:
            self._errors.append(key + "：当前LSTM配置不支持门值重建")
            return
        current = x[0]
        all_gates = {name: [] for name in ("i", "f", "g", "o")}
        max_error = 0.0
        for index in range(module.num_layers):
            gates = (current @ getattr(module, f"weight_ih_l{index}").T
                     + h0[index] @ getattr(module, f"weight_hh_l{index}").T)
            if module.bias:
                gates = gates + getattr(module, f"bias_ih_l{index}") + getattr(module, f"bias_hh_l{index}")
            i, f, g, o = gates.chunk(4, dim=-1)
            i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
            reconstructed_c = f * c0[index] + i * g
            current = o * reconstructed_c.tanh()
            error = max(float((current - h[index]).abs().max()), float((reconstructed_c - c[index]).abs().max()))
            max_error = max(max_error, error)
            for name, value in zip(all_gates, (i, f, g, o)):
                all_gates[name].append(value[0])
        if max_error > 2e-5:
            self._errors.append(f"{key}：门值重建误差{max_error:.3g}，已隐藏门图")
            return
        for name, values in all_gates.items():
            self._save(key + "." + name, torch.stack(values), strip_batch=False)

    def _pre_hook(self, key, _module, inputs):
        if self._phase != "actor":
            return
        if key == "features_extractor":
            for name, tensor in inputs[0].items():
                self._save("input." + name, tensor)

    def _post_hook(self, key, module, inputs, output):
        actor_keys = not (key.startswith("lstm_critic") or key.startswith("mlp_extractor.value_net") or key == "value_net")
        if (self._phase == "actor") != actor_keys:
            return
        if isinstance(module, nn.LSTM):
            self._lstm(key, module, inputs, output)
        else:
            self._save(key, output)
        if key == "features_extractor.boxes":
            masked = output * inputs[0][:, :, 8:9]
            self._save("detections.masked", masked)
            self._save("detections.pooled", masked.amax(dim=1))
            self._save("detections.winner", masked.argmax(dim=1))

    def _safe_hook(self, function, *args):
        try:
            function(*args)
        except Exception as exc:
            self._errors.append(f"采集失败：{type(exc).__name__}: {exc}")

    def _critic(self, observation, starts):
        lstm = getattr(self.policy, "lstm_critic", None)
        if not isinstance(lstm, nn.LSTM):
            self._errors.append("该检查点没有独立价值LSTM；价值旁路未采集")
            return
        if self.critic_state is None:
            shape = (lstm.num_layers, 1, lstm.hidden_size)
            self.critic_state = (torch.zeros(shape, device=self.policy.device), torch.zeros(shape, device=self.policy.device))
        tensor_obs, _ = self.policy.obs_to_tensor(observation)
        self.policy.predict_values(tensor_obs, self.critic_state,
                                   torch.as_tensor(starts, dtype=torch.float32, device=self.policy.device))

    def _capture(self, observation, sim_time: float, *, preview: bool):
        if int(getattr(self.runner, "num_envs", 1)) != 1:
            raise ValueError("模型查看器仅采集单环境；不聚合多个车辆的激活")
        starts = self.runner.episode_start.copy()
        old_critic = self.critic_state
        if starts.any():
            self.critic_state = None
        for layer in self.layers.values():
            layer["values"] = None
        self._errors = []
        self._phase = "actor"
        handles = []
        cache_modes = []
        for key, module in self.modules.items():
            if hasattr(module, "cache_inference"):
                cache_modes.append((module, module.cache_inference))
                module.cache_inference = False
            if key in self.layers or key == "features_extractor.boxes":
                handles.append(module.register_forward_hook(
                    lambda module, inputs, output, key=key: self._safe_hook(self._post_hook, key, module, inputs, output)))
            if key == "features_extractor":
                handles.append(module.register_forward_pre_hook(
                    lambda module, inputs, key=key: self._safe_hook(self._pre_hook, key, module, inputs)))
        try:
            with torch.no_grad():
                if preview:
                    # predict returns fresh actor states; never assign them to runner.
                    state = None if self.runner.state is None else tuple(np.array(value, copy=True) for value in self.runner.state)
                    action, _ = self.runner.model.predict(observation, state=state, episode_start=starts, deterministic=True)
                else:
                    action = self.runner.action(observation)
                self._refresh_runtime_dimensions()
                self._save("action.applied", torch.as_tensor(action))
                log_std = getattr(self.policy, "log_std", None)
                self._save("action.log_std", log_std, strip_batch=False)
                distribution = getattr(self.policy.action_dist, "distribution", None)
                if distribution is not None and hasattr(distribution, "stddev"):
                    self._save("action.std", distribution.stddev)
                self._phase = "critic"
                try:
                    self._critic(observation, starts)
                except Exception as exc:
                    self._errors.append(f"价值旁路不可用：{type(exc).__name__}: {exc}")
        finally:
            for handle in handles:
                handle.remove()
            for module, mode in cache_modes:
                module.cache_inference = mode
            if preview:
                self.critic_state = old_critic
        self.sequence += 1
        self.snapshot = dict(sequence=self.sequence, action_config=action_config(), sim_time=float(sim_time),
                             mode="暂停预览（未施加车辆）" if preview else "实际驾驶推理（动作前观测）",
                             critic="价值支路为独立记忆的额外诊断计算；记忆自本段连续采集起累计，不控制车辆",
                             errors=tuple(self._errors), layers=list(self.layers.values()))
        return action

    def action(self, observation, sim_time: float):
        return self._capture(observation, sim_time, preview=False)

    def preview(self, observation, sim_time: float):
        self._capture(observation, sim_time, preview=True)
        return self.snapshot
