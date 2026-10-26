# 按键预测 20 神经元

这是一个固定 20 个活动单元的本地在线训练原型。用户每按一次 0–9，服务先用先前封存的预测计分并写入记录，再把实际数字加入训练历史，训练后才生成下一次预测。它是一个可塑 RNN 起点，不声称已经自动发现或验证了通用学习算法。

## 运行

项目默认使用已安装 CPU 版 PyTorch 的环境：

```powershell
& 'C:\pioneer\mnist-local-neuron-env\Scripts\python.exe' '.\server.py' --host localhost --port 8765
```

也可用启动脚本：

```powershell
& '.\start.ps1'
```

脚本默认 Python 路径为 `C:\pioneer\mnist-local-neuron-env\Scripts\python.exe`，可通过 `-Python` 指定其他解释器：

```powershell
& '.\start.ps1' -Python 'C:\path\to\venv\Scripts\python.exe' -Port 8765
```

若需新建 Python 环境，先安装依赖：

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --index-url https://download.pytorch.org/whl/cpu -r '.\requirements.txt'
& '.\.venv\Scripts\python.exe' '.\server.py' --host localhost --port 8765
```

`requirements.txt` 固定 PyTorch `2.14.1`；项目运行环境已验证为 `2.14.1+cpu`。

服务只监听 `127.0.0.1`，`--host` 只接受 `localhost`。网页由相邻的 `dist/` 目录提供；API 可单独使用。默认端口是 `8765`。

数据写在版本库外。默认每次启动都会在 `C:\pioneer\local-neuron-runs\` 下创建唯一的 `keypress20_<时间>_<随机后缀>` 目录；传入 `--data-root` 可更换其父目录。每个会话有独立 UUID 子目录。按“重开会话”会创建新子目录，不清除旧日志；重启进程也会创建一轮全新的模型和 run 目录。

## 模型与训练

输入端口是 10 个数字的 one-hot 向量，不计入 20 个神经元。每一步读取旧活动 `h` 和旧边迹 `F`，按以下公式更新活动：

```text
h' = tanh(Win · x_digit + (Wrec + eta · F) · h + b)
```

`Wrec` 是完整的 `20 × 20` 有向矩阵。矩阵行是目标单元、列是来源单元，因此 `connections[i][j]` 表示从单元 `j` 到单元 `i` 的连接。没有出边数量上限。活动更新后，局部边迹按来源/目标活动更新：

```text
F' = decay · F + (1 - decay) · outer(h', h)
```

`eta` 与 `decay` 是共享可训练标量。输出端口直接取前 10 个活动值乘以一个共享正比例作为 logits，再做 softmax；模型没有额外的 10 个读出单元。

每次计分后，模型用最近最多 32 个已发生的数字构成相邻预测对，做 3 次 Adam 更新，每次展开不超过 32 个观察步。之后从零状态重放该上下文，得到下一次预测。超出 32 个数字的更早上下文不进入当前状态，因此模型不被描述为具有无限记忆。没有可训练转移对时预测为均匀分布；启动时并列 top-1 取数字 0，所以首次实际输入会对该封存的 0 选项记分。训练来源随响应标成 `uniform_cold_start` 或 `plastic_rnn`。

## 实时结构图

主界面的结构图显示固定 20 个神经元及其有向循环连接。节点位置仅用于排布，不代表网络分层；第 1–10 个神经元同时对应数字 0–9 的输出端口。

节点颜色按固定的 `[-1, 1]` 活动范围显示正负方向，颜色深浅表示绝对值大小。每次输入完成后，图与本轮训练后的活动、连接权重和按键步数同步更新；没有新输入时模型状态保持不变。图展示上下文重放完成后的状态快照，不展示展开计算中每一步的传播动画。

连接箭头从来源神经元指向目标神经元，权重使用 API 返回的有效连接 `Wrec + eta * F`。连接筛选只影响图中显示，模型仍保留完整的 `20 × 20` 连接。选中节点可查看活动数值及与其有关的连接。

窄屏默认显示完整结构图；点击“放大查看”可横向浏览放大的节点和连接，点击“显示全图”恢复总览。

刷新页面即可加载新版界面，正在运行的服务和当前训练会话无需重启。

## API

- `GET /api/state` 返回当前会话、封存预测 ID、概率、预测来源、准确率指标、20 个活动、`20 × 20` 连接矩阵、最近 30 个历史数字，以及上一条 `last_result`。
- `POST /api/press` 接收 JSON `{"digit": 0, "prediction_id": "..."}`。`digit` 必须是 JSON 整数且严格在 0–9 内；布尔值、浮点、字符串均会拒绝。成功响应包含新的 state 和被消费预测对应的 `last_result`。同一预测 ID 只能消费一次；过期或重复 ID 返回 HTTP 409，错误形如 `{"error":{"code":"stale_prediction","message":"..."}}`。请求体上限为 4096 字节。
- `POST /api/reset` 可不带 body 或发送 `{}`，创建全新的会话与模型状态，旧日志仍留在 run 目录。
- `GET /api/export` 下载 `application/json` 文件，包含 run 下所有会话清单及 JSONL 事件。

错误统一使用 `{"error":{"code":"...","message":"..."}}`。HTTP API 的 JSON 响应不缓存。

## 记录与范围

每次启动会在 run 目录中保存 `source_snapshot/`，逐字节复制模型、服务、依赖清单、README、启动脚本和可用的 `dist/` 文件，并写入包含各文件 SHA-256 的清单。run 与会话清单还记录配置 SHA-256、Python/PyTorch 版本和线程数。`events.jsonl` 每次按键写两条按 `session_id` 与步骤关联的记录：第一条在历史更新前保存封存概率、选择、实际数字和得分；第二条在训练完成后保存训练耗时、可塑参数值和下一预测、活动与连接。这种顺序便于重放核对计分没有使用当前真实数字重算。

每次按键训练和新预测完成后，会把模型、Adam 状态、最近历史、计数、封存的下一预测等写入 `latest_checkpoint.pt.tmp`，完成后通过同目录原子替换更新 `latest_checkpoint.pt`。目前该检查点供审计与后续恢复功能使用，程序重启仍从新 session 开始。

`events.jsonl` 和清单会保留；当前版本不恢复训练检查点，停止后重新运行会开始新的 run。准确率是本模型 top-1 的累计准确率；随机基线固定为 10%，多数类基线每次只根据当时以前输入的数字计算，平票选最低数字。
