# 目录文件索引

当前目录提供 LAN worker 与状态拉取辅助。worker_agent.py 的源码启动的是 V1.1 runner；它不启动 Native V2 正式配方。V2 正式训练入口见 ../train/README.md。跨机命令和可视化步骤如下文所述，实际协议状态以 ../MIGRATION_ACCEPTANCE.md 为准。

| 文件 | 用途 |
| --- | --- |
| pull_live_state.py | 按 token 轮询远端 worker 的 /live_state endpoint，并原子更新本机状态 JSON，供现有 Viser viewer 读取。 |
| worker_agent.py | 提供带 X-Auth-Token 请求头的 LAN HTTP worker：启动/停止 V1.1 trial，读取状态、指标、live-state 和有限的 run 文件。 |

# 远程执行与换机（SSH + Git + rsync）

目标：本机（协调者/DSH Agent）驱动另一台 GPU 机器跑 native trial，并回看日志、checkpoint 和可视化。
**不需要 A2A**；A2A 只适用于"不同 Agent 框架之间的语义任务委派"，跨机跑训练用下面这套即可。

## 1. 网络与登录

```bash
# 机器 B（训练机）安装并启动 OpenSSH Server
# 机器 A 生成密钥（若还没有）
ssh-keygen -t ed25519 -C "dsh-agent"
ssh-copy-id user@machine-b        # 或手动把公钥写入 B 的 ~/.ssh/authorized_keys

# 跨网络/无公网 IP：两边都装 Tailscale，使用 tailscale IP
tailscale ip -4
ssh user@100.x.y.z
```

约定：机器 B 的仓库路径固定为 `$REMOTE_REPO`（例如 `/data/AGTC`），WSL 路径为
`/mnt/c/pioneer/先进通用履带底盘工程`；下文的命令都通过环境变量传入，不把密码/密钥写进仓库。

## 2. 机器 A 上的常用命令

```bash
# 同步代码（B 首次 clone，之后 pull）
ssh user@B "git clone https://github.com/w3743/AGTC $REMOTE_REPO && cd $REMOTE_REPO && git checkout perf/max-throughput"

# 启动正式 trial（后台，写 manifest/metrics）
ssh user@B "cd $REMOTE_REPO/算法/仿真训练环境/native && \
  nohup env SIM_DT=0.001 DECIMATION=50 bash train/run_native_v1_trial.sh pilot02 \
    --live-state-file /tmp/native_trial_live.json --live-state-interval 5 \
    > /tmp/pilot02.log 2>&1 & echo \$!"

# 轮询
ssh user@B "tail -n 5 $REMOTE_REPO/算法/仿真训练环境/native/runs/native_v1_1_pilot02_*/metrics.jsonl"

# 拉回结果（日志/指标/审计 JSON；checkpoint 视需要）
rsync -av user@B:$REMOTE_REPO/算法/仿真训练环境/native/runs/native_v1_1_pilot02_*/ \
  ./remote_results/pilot02/

# 停止（按 manifest 里的 PID 或 pkill）
ssh user@B "pkill -f run_native_v1_trial.sh; pkill -f train_wsl.py"
```

## 3. 看 B 上的可视化（SSH 端口转发）

```bash
# 在 B 上启动查看器（中文界面，监听 127.0.0.1:8792）
ssh user@B "cd $REMOTE_REPO/算法/仿真训练环境 && \
  python scripts/view_training_state_web.py --state-file /tmp/native_trial_live.json \
    --output /tmp/viewer_status --port 8792 --language zh &"

# 在 A 上建立隧道，然后本机浏览器打开 http://127.0.0.1:8792
ssh -N -L 8792:127.0.0.1:8792 user@B
```

## 4. 最小协调协议（如果两边都跑 Agent）

- 共享目录或 Git 分支中放：`task.json`（配置、协议 SHA、seed）、`status.json`（阶段、iter、PID）、
  `result.json`（指标、evaluation 路径）；
- 两边 Agent 轮询/更新这些文件即可，语义清晰、可审计；
- 需要更强控制时，在 B 上起一个只监听 127.0.0.1 的 HTTP 控制面（`/start`、`/status`、`/stop`），
  用 SSH 隧道访问，仍然不需要 A2A。

## 5. 安全

- 只用密钥登录，禁用密码；
- 不要把 GitHub token、SSH 私钥、W&B key 写进仓库或命令行历史；
- 端口只通过 SSH 隧道访问，不对公网监听；
- 结果目录里的 checkpoint 不进 Git，用 rsync 同步。

## 6. 局域网（LAN）专用说明

同一交换机/路由器下最简单：**直连内网 IP，不需要 Tailscale、不需要 A2A**。

### 6.1 先确认地址与连通

```bash
# 机器 B（训练机）
ip a            # Linux：找 192.168.x.x/10.x.x.x
ipconfig        # Windows：IPv4 地址
hostname -I     # Linux
# 机器 A 上验证
ping 192.168.1.50
```

建议在路由器上给 B 做 **DHCP 地址保留**（固定 IP），避免换机后地址变化；或在 B 上配置静态 IP。
Windows 下主机名可用 `ping <hostname>`（NetBIOS）；Linux 可用 `hostname.local`（mDNS，需 avahi）。

### 6.2 方案一：SSH（第二台机器是 Linux，或有管理员权限装 OpenSSH）

```bash
# B（Linux）
sudo apt update && sudo apt install -y openssh-server
sudo systemctl enable --now ssh
sudo ufw allow from 192.168.0.0/16 to any port 22 proto tcp   # 如果启用了 ufw

# A
ssh-copy-id user@192.168.1.50
ssh user@192.168.1.50 'nvidia-smi; python -V'
```

之后按本文件 §2–§3 的模板执行：启动 trial、轮询 metrics、rsync 拉结果、`ssh -N -L 8792:127.0.0.1:8792 user@B` 看 Viser。

Windows 机器 B 若已启用 OpenSSH Server（`Get-Service sshd`），也可直接用 SSH；没有管理员权限时用方案二。

### 6.3 方案二：HTTP Worker（无需管理员，Linux/Windows/WSL 都行）

机器 B 上（任意 Python 3.10+，仅标准库）：

```bash
# B：在 AGTC 仓库根目录
set AGTC_WORKER_TOKEN=<随机长字符串>        # PowerShell；Linux: export AGTC_WORKER_TOKEN=...
python 算法/仿真训练环境/native/remote/worker_agent.py \
    --host 0.0.0.0 --port 8790 --repo <仓库根目录>
# 首次运行时 Windows 防火墙会弹窗，选择“允许访问”（专用网络即可；无需管理员）
```

机器 A 上：

```powershell
$h=@{ 'X-Auth-Token' = '<随机长字符串>' }
Invoke-RestMethod -Uri 'http://192.168.1.50:8790/status' -Headers $h

# 启动正式 trial（1ms/50）
$body = @{ tag='pilot02'; sim_dt=0.001; decimation=50; live_state_file='/tmp/native_trial_live.json'; live_state_interval=5 } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://192.168.1.50:8790/start' -Headers $h -Method Post -ContentType 'application/json' -Body $body

# 轮询 / 停止 / 看 metrics
Invoke-RestMethod -Uri 'http://192.168.1.50:8790/status' -Headers $h
Invoke-RestMethod -Uri 'http://192.168.1.50:8790/metrics?lines=20' -Headers $h
Invoke-RestMethod -Uri 'http://192.168.1.50:8790/stop' -Headers $h -Method Post
```

实时可视化（A 上运行，仍是原有 Viser 查看器）：

```powershell
# 持续拉取 B 的 live_state 到本地文件
python 算法/仿真训练环境/native/remote/pull_live_state.py `
    --url http://192.168.1.50:8790 --token <随机长字符串> `
    --out C:/pioneer/wsl-native/live/native_live_state.json
# 另开一个进程运行本地查看器
python 算法/仿真训练环境/scripts/view_training_state_web.py `
    --state-file C:/pioneer/wsl-native/live/native_live_state.json `
    --output C:/pioneer/wsl-native/live/viewer_status --port 8792 --language zh
# 浏览器打开 http://127.0.0.1:8792
```

安全注意：token 只放环境变量/命令行；worker 只绑定局域网（不要做公网端口映射）；不需要时停掉进程。

### 6.4 选型建议

| 第二台机器 | 推荐 |
|---|---|
| Linux（裸机或 WSL2 mirrored） | **SSH**（最简单） |
| Windows 且无管理员权限 | **HTTP Worker**（方案二，已提供并本地自测通过） |
| 需要跨公网/无公网 IP | Tailscale + 上面任一方案 |
