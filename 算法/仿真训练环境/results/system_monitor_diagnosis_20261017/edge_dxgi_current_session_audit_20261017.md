# Edge GPU preference and adapter-LUID audit

## Current read-only evidence

The current per-user Edge value is `HKCU:\Software\Microsoft\DirectX\UserGpuPreferences` → `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe` = `GpuPreference=2;`. This confirms the saved high-performance preference; no registry value was changed. The corresponding HKLM key was absent.

The project search used `rg -n -i --glob '!external/**' --glob '!results/**' 'DXGI|AdapterLuid|LUID|UserGpuPreferences|EnumAdapter|GetDesc1' .` and a filename search over `scripts`, `external`, and `results`. No reusable source script or implementation that enumerates DXGI adapters and maps LUIDs to adapter descriptions was found. The existing `results/system_monitor_diagnosis_20261017/collect_windows_perf.ps1` collects raw GPU Engine LUIDs and process names but intentionally does not map an LUID to an adapter.

Historical records are not a current-session mapping: `results/isaac_field/edge_gpu_verified.json` associates the NVIDIA process observation for old Edge PID 29788 with LUID `0x00000000_0x0001978F` at 18:18; `results/gazebo_web/gpu_runtime_diagnostic_20261017.json` explicitly left the physical adapter for LUID `0x00000000_0x00018384` unknown. Current `Win32_VideoController` names confirm that Intel and NVIDIA adapters are present, but do not bind either name to an Engine LUID.

## Pending same-session check

At the time of this inspection the main agent had started a viewer on port 8768, but had not yet opened the browser page. No Edge Engine sample was taken at that point. After the page is opened, correlate the live Edge GPU PID and its raw GPU Engine LUID with the same-session `nvidia-smi` graphics process table. This can positively identify an Edge LUID as NVIDIA when the same PID appears in both observations. If no such match occurs, report the mapping as unverified; do not infer Intel from adapter inventory or the saved preference.

Commands used: read the Edge-specific registry value with `Get-Item HKCU:\Software\Microsoft\DirectX\UserGpuPreferences` and `GetValueNames()/GetValue()` filtered only to `msedge.exe`; search project source/results with the `rg` commands above. No setting, Edge process, browser page, or driver was changed or restarted.
