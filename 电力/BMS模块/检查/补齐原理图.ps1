$ErrorActionPreference='Stop'
$edaCli='D:\lceda-pro\lceda-pro.exe'
$edaSession='d89c487f-db54-4a5e-a326-d10278f67ad2'
$items=@(Get-Content -LiteralPath 'BMS模块\检查\原理图构建清单.json' -Raw|ConvertFrom-Json)
$bmsMap=Get-Content -LiteralPath 'BMS模块\检查\原理图页面映射.json' -Raw|ConvertFrom-Json
$groups=@($items.pageGroup|Select-Object -Unique)
$missing=@($groups|Where-Object {!$bmsMap.PSObject.Properties[$_]})
$payload=ConvertTo-Json -InputObject $missing -Compress
$code="const groups=$payload;const out={};let first=true;for(const n of groups){const p=first?'02679108d4293745':await eda.dmt_Schematic.createSchematicPage('42dbb456fb4dd67a');first=false;await eda.dmt_Schematic.modifySchematicPageName(p,n);out[n]=p;}return out;"
$r=(& $edaCli invoke --session $edaSession --ext-uuid eda --timeout 60000 --code $code|Out-String|ConvertFrom-Json)
if(!$r.ok){throw($r|ConvertTo-Json -Depth 10)}
foreach($p in $r.value.PSObject.Properties){$bmsMap|Add-Member -NotePropertyName $p.Name -NotePropertyValue $p.Value}
$bmsMap|ConvertTo-Json|Set-Content -LiteralPath 'BMS模块\检查\原理图页面映射.json' -Encoding utf8
foreach($n in @('AFE_U1_CELL_FILTERS','AFE_CELL_TAPS','AFE_BREG_HOST')+$missing){
 $page=$bmsMap.$n;$subset=@($items|Where-Object pageGroup -eq $n)
 $path=Join-Path 'BMS模块\检查' ('分组-'+$n+'.json')
 ConvertTo-Json -InputObject $subset -Depth 30|Set-Content -LiteralPath $path -Encoding utf8
 & 'BMS模块\检查\原理图构建.ps1' -Manifest $path -Page $page -Count 1000|Out-Null
 $last=Get-Content -LiteralPath ('BMS模块\检查\构建-'+$page+'-0.json') -Raw|ConvertFrom-Json
 [pscustomobject]@{group=$n;page=$page;ok=$last.ok;count=$last.value.components.Count;error=$last.error}|ConvertTo-Json -Depth 8 -Compress|Write-Output
 if(!$last.ok){break}
}
