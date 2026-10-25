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
