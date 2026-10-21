[CmdletBinding()]
param(
    [ValidateSet('Install','Start','Check')][string]$Mode = 'Start',
    [string]$Distribution,
    [switch]$NoPause
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$projectPath = Split-Path $PSScriptRoot -Parent
$settingsPath = Join-Path $projectPath '.local.json'
$settings = if (Test-Path -LiteralPath $settingsPath) { Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json } else { [pscustomobject]@{ distribution='Ubuntu-24.04' } }
if (-not $Distribution) {
    $Distribution = $settings.distribution
}
$transcript = $false
$code = 0
function Invoke-Linux([string[]]$LinuxArgs, [switch]$AsRoot) {
    $baseArgs = @('-d', $Distribution)
    if ($AsRoot) { $baseArgs += @('-u', 'root') }
    & wsl.exe @baseArgs --exec @LinuxArgs | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "WSL 操作失败（退出码 $LASTEXITCODE）。请查看上方错误。" }
}
try {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { throw '请先安装 WSL2：以管理员身份运行 wsl --install -d Ubuntu-24.04，重启后首次打开 Ubuntu 创建用户。' }
    $kernel = & wsl.exe -d $Distribution --exec uname -r
    if ($LASTEXITCODE -ne 0) { throw "没有可用的 $Distribution。以管理员身份运行 wsl --install -d $Distribution，按提示重启并在 Ubuntu 中创建用户，然后重新双击安装环境.cmd。" }
    if ($kernel -notmatch 'WSL2') { throw "此发行版不是 WSL2。请先执行：wsl --set-version $Distribution 2" }
    $linuxUser = (& wsl.exe -d $Distribution --exec id -un).Trim()
    if ($linuxUser -eq 'root') { throw 'Ubuntu 默认用户是 root。请先创建并设置普通 Linux 用户，再运行安装器。' }
    $linuxProject = (& wsl.exe -d $Distribution --exec wslpath -u $projectPath.Replace('\','/')).Trim()
    if (-not $linuxProject) { throw '无法访问解压目录。请放在本地磁盘，不要从压缩包内或网络共享直接运行。' }
    New-Item -ItemType Directory -Force -Path (Join-Path $projectPath 'results') | Out-Null
    if ($Mode -eq 'Install') {
        Start-Transcript -Path (Join-Path $projectPath 'results\install.log') -Force | Out-Null
        $transcript = $true
        Write-Host '安装 ROS/Gazebo、固定版本 Python 环境及 PX4/UWB。需要联网，首次安装可能较久。' -ForegroundColor Cyan
        Invoke-Linux -AsRoot -LinuxArgs @('bash', "$linuxProject/scripts/install.sh", 'system')
        Invoke-Linux -LinuxArgs @('bash', "$linuxProject/scripts/install.sh", 'user')
        $settings | Add-Member -NotePropertyName distribution -NotePropertyValue $Distribution -Force
        $settings | ConvertTo-Json | Set-Content -LiteralPath $settingsPath -Encoding UTF8
        Write-Host '安装和基础检查完成。请双击 启动实验室.cmd。' -ForegroundColor Green
    } elseif ($Mode -eq 'Check') {
        Invoke-Linux -LinuxArgs @('bash', "$linuxProject/scripts/launch.sh", 'check')
    } else {
        Write-Host "`n机器人虚拟实验室" -ForegroundColor Cyan
        Write-Host '1  无人车：视觉建图 + UWB（自动巡航3分钟）'
        Write-Host '2  无人车：手动实验'
        Write-Host '3  无人机：自动起飞、悬停、降落'
        Write-Host '4  机械臂：模型回放（没有模型则演示随机动作）'
        Write-Host '5  灵巧手：模型回放'
        Write-Host '6  机械臂：短训练4096步'
        Write-Host '7  灵巧手：短训练4096步'
        Write-Host '8  小车键盘控制（先启动2）'
        Write-Host '9  查看说明和结果'
        Write-Host '10 LightStereo：虚拟双目深度与真值对比（5个视角）'
        Write-Host '11 LightStereo：减少黑色区域的对照实验（5+5个视角）'
        Write-Host '12 距离精度研究：RAFT / FoundationStereo 对比（Windows GPU，追加5个新视角）'
        Write-Host '13 距离精度改进：多模型互验与输入尺度对照（需要先完成12）'
        Write-Host '14 低纹理原因验证：均匀材质 / 地面纹理 / 全场景纹理对照'
        Write-Host '15 LightStereo独立复测：原输入 / Gamma，同一批原始与纹理场景（需要已有13、14结果）'
        Write-Host '0  退出'
        $selection = Read-Host '请输入编号'
        $tasks = @{'1'='mapping';'2'='manual';'3'='drone';'4'='arm-view';'5'='hand-view';'6'='arm-train';'7'='hand-train';'8'='teleop';'10'='lightstereo';'11'='lightstereo-study'}
        if ($selection -eq '9') {
            Invoke-Item (Join-Path $projectPath '使用说明.md')
            Invoke-Item (Join-Path $projectPath 'results')
        } elseif ($selection -eq '0') { $NoPause = $true }
        elseif ($selection -in @('12','13','14','15')) {
            $stereoCache = if ($settings.stereo_cache) { $settings.stereo_cache } else { Join-Path $env:LOCALAPPDATA 'stereo-lab' }
            $stereoPython = if ($settings.stereo_python) { $settings.stereo_python } else { Join-Path $stereoCache 'venv\Scripts\python.exe' }
            if (-not (Test-Path -LiteralPath $stereoPython)) { throw '缺少 Windows GPU 实验环境。请按使用说明中的距离精度研究配置 stereo_python 和 stereo_cache。' }
            if ($selection -ne '15') {
                foreach ($relativePath in @('RAFT-Stereo\models\raftstereo-middlebury.pth','RAFT-Stereo\models\iraftstereo_rvc.pth','FoundationStereo\pretrained_models\11-33-40\model_best_bp2.pth')) {
                    if (-not (Test-Path -LiteralPath (Join-Path $stereoCache $relativePath))) { throw "缺少实验权重：$relativePath。请查看使用说明。" }
                }
            }
            $env:PYTHONIOENCODING = 'utf-8'
            $entry = if ($selection -eq '12') { 'simulation\compare_stereo.py' } else { 'simulation\refine_stereo.py' }
            $extraArgs = if ($selection -eq '14') { @('--appearance') } else { @() }
            if ($selection -eq '15') {
                & $stereoPython -B (Join-Path $projectPath $entry) --lightstereo | Out-Host
            } else {
                & $stereoPython -B (Join-Path $projectPath $entry) --cache $stereoCache --capture --distribution $Distribution @extraArgs | Out-Host
            }
            if ($LASTEXITCODE -ne 0) { throw "距离精度实验失败（退出码 $LASTEXITCODE）。请查看上方错误。" }
        }
        elseif ($tasks.ContainsKey($selection)) { Invoke-Linux -LinuxArgs @('bash', "$linuxProject/scripts/launch.sh", $tasks[$selection]) }
        else { throw '编号无效，请重新启动。' }
    }
} catch {
    $code = 1
    Write-Host "`n$($_.Exception.Message)" -ForegroundColor Red
} finally {
    if ($transcript) { Stop-Transcript | Out-Null }
    if (-not $NoPause) { [void](Read-Host '按 Enter 关闭') }
}
exit $code
