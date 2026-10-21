# GPU correctness health check

This was a bounded correctness check, not a simulator run or performance benchmark. Before running it, Windows `nvidia-smi` reported RTX 5070 Ti at 0% GPU utilization, 217/12227 MiB, 51 °C. Ubuntu-24.04 and Docker Desktop were stopped, and the Windows process scan found no Isaac training or viewer candidates. The check then started the stopped Ubuntu distro normally through the existing project wrapper; no WSL, driver, or computer reset was issued. Ubuntu remained running afterward.

The command was `./scripts/isaaclab_python.sh results/isaac_field_diagnosis/stem_contact_v2_20261017/gpu_health_recovery_20261017/gpu_health_probe.py`, invoked from Ubuntu-24.04 at the project root. Wrapper/WSL exit code was **0**, and the probe result itself was **ok: true**. It used CUDA explicitly; no CPU fallback was accepted.

The installed runtime reported Python 3.12.3 on WSL2 (`Linux 6.6.87.2-microsoft-standard-WSL2`), PyTorch 2.11.0+cu130, CUDA 13.0, cuDNN 9.19.0 (version integer 91900), and Warp 1.17.0. Both frameworks detected an NVIDIA GeForce RTX 5070 Ti Laptop GPU, compute capability 12.0; Warp initialized CUDA Toolkit 12.9 with Driver 13.4.

All three synchronized checks passed: a CUDA tensor calculation and device-to-host copy, cuDNN Conv2d forward/backward with finite output and gradients, and a Warp CUDA kernel followed by its array-to-CPU copy. The WSL post-check reported 0% utilization and 120/12227 MiB on the NVIDIA device at 52 °C; no Python/Isaac/viewer process names appeared in its filtered process listing.

The WSL startup output included a localized warning mentioning localhost/NAT whose text was partly garbled by PowerShell's capture encoding; a post-check shell message also said `your 131072x1 screen size is bogus. expect trouble`. These did not fail the wrapper or the GPU checks. No CUDA error 999 appeared in this bounded check. The original wrapper output is preserved verbatim in [wsl_wrapper_output.log](wsl_wrapper_output.log); see [probe_result.json](probe_result.json), [wrapper_execution.json](wrapper_execution.json), and [wsl_postcheck.log](wsl_postcheck.log) for structured results and status.
