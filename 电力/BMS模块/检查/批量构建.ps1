$ErrorActionPreference='Stop'
$edaCli='D:\lceda-pro\lceda-pro.exe';$edaSession='d89c487f-db54-4a5e-a326-d10278f67ad2'
$items=@(Get-Content -LiteralPath 'BMS模块\检查\原理图构建清单.json' -Raw|ConvertFrom-Json)
$names=@($items.pageGroup|Select-Object -Unique)
$payload=ConvertTo-Json -InputObject $names -Compress
$code="const names=$payload;const existing={AFE_U1_CELL_FILTERS:'da5473602d719185',PWR_MAIN_PATH:'82222f7cb9045883',IF_PSU_POWER:'79c5ec31cd804dea',IF_CAN_ISOLATION:'a35cf8773708333e'};const out={};for(const n of names){const p=existing[n]??await eda.dmt_Schematic.createSchematicPage('42dbb456fb4dd67a');await eda.dmt_Schematic.modifySchematicPageName(p,n);out[n]=p;}return out;"
$r=(& $edaCli invoke --session $edaSession --ext-uuid eda --timeout 60000 --code $code|Out-String|ConvertFrom-Json)
if(!$r.ok){throw($r|ConvertTo-Json -Depth 10)}
$r.value|ConvertTo-Json|Set-Content -LiteralPath 'BMS模块\检查\原理图页面映射.json' -Encoding utf8
foreach($n in $names){$page=$r.value.$n;$subset=@($items|Where-Object pageGroup -eq $n);$path=Join-Path 'BMS模块\检查' ('分组-'+$n+'.json');ConvertTo-Json -InputObject $subset -Depth 30|Set-Content -LiteralPath $path -Encoding utf8;$out=(& 'BMS模块\检查\原理图构建.ps1' -Manifest $path -Page $page -Count 1000|Out-String);$last=Get-Content -LiteralPath ('BMS模块\检查\构建-'+$page+'-0.json') -Raw|ConvertFrom-Json;[pscustomobject]@{group=$n;page=$page;ok=$last.ok;count=$last.value.components.Count;error=$last.error.message}|ConvertTo-Json -Compress|Write-Output}
