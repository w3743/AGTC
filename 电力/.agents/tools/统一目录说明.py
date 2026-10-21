"""整理后更新当前入口，历史链接指向已校验归档；不改电气设计或模型。"""
from pathlib import Path
import json, re, os, zipfile, hashlib
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[2]
receipt=json.loads((ROOT/'归档/目录整理.json').read_text(encoding='utf-8'))
if receipt['status']!='VERIFIED_READY_FOR_POWERSHELL_CLEANUP' or not all(x['readback'] for x in receipt['files']):raise RuntimeError('归档未完成读回')
def file_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
for rel,info in receipt['archives'].items():
    if file_sha(ROOT/rel)!=info['after_sha256']:raise RuntimeError('已读回ZIP变化: '+rel)
removed={x['path']:x for x in receipt['files'] if x.get('cleanup',True)}
if any((ROOT/rel).exists() for rel in removed):raise RuntimeError('请先按已校验清单清理原件和空目录')
document_backups=[]

def put(rel,text):
    (ROOT/rel).write_text(text.strip()+'\n',encoding='utf-8')

put('README.md','''# 电力工程

每块板只保留一份当前原生工程和一套当前交付。后续原位更新，历史和调试材料归档。

| 模块 | 当前入口 | 当前检查 |
|---|---|---|
| FOC驱动模块（两块同版） | [模块说明](FOC驱动模块/README.md) · [原生工程](FOC驱动模块/eda工程文件/FOC_G431_Driver/FOC_G431_Driver.eprj3) | [原理图复核](FOC驱动模块/检查/原理图电气复核结论.md) |
| 稳压模块（超容双向DC/DC） | [模块说明](稳压模块/README.md) · [原生工程](稳压模块/eda工程文件/sc-regulator/sc-regulator.eprj3) | [原理图复核](稳压模块/检查/原理图复核结论.md) |
| BMS模块（9S4P） | [项目要求](BMS模块/项目要求.md) · [模块说明](BMS模块/README.md) · [原生工程](BMS模块/BMS-9S4P.eprj3) | [当前状态与未决项](BMS模块/检查/当前设计状态.md) |

共同设计见[基本设计](基本设计.md)、[规则](规则.md)；选型与封装统一从[元件封装库](元件封装库/README.md)进入。代理约定见[AGENTS](AGENTS.md)。

## 统一目录约定

| 位置 | 保留内容 |
|---|---|
| 模块根 | README、当前要求、TODO及现行设计说明 |
| `eda工程文件/` | 一份当前原生工程及当前交付；BMS按用户要求例外，以模块根EPRJ3和配套`sch/`为唯一当前入口 |
| `原理图准备/` | 现行选型、BOM设计输入、预期连接、接口和Layout约束 |
| `检查/` | 支持当前结论的证据与必要复核工具 |
| `仿真/源码/`、`输入/`、`结果/` | 现行可复现模型、输入与当前结果；必要运行时原位保留 |
| `参考资料/` | 厂商、元件与开源参考资料；BMS已有的`厂商资料/`原位保留以维持引用 |
| `归档/历史与调试.zip` | 该模块旧方案、过程脚本、旧报告和调试数据 |

全项目共用过程材料集中到[工程过程.zip](归档/工程过程.zip)。本轮归档映射、逐文件SHA256读回凭据及正式文件不变检查见[目录整理记录](归档/目录整理.json)。本轮新增归档成员使用电力根目录相对路径；旧ZIP成员保持原命名，恢复时按记录选择，勿统一假定为模块根路径。

`归档/原始材料/`仅保留当前验证实际依赖或尚不能安全清理的材料，不作为当前工程入口。项目技能、配置及共用库Git工作树原位保留。

机器证据与冻结输入快照保留原始历史路径供溯源，恢复对应材料时查整理记录。根`tmp/`中的旧公共库导出已归档；需要准备下一次发布时运行共用库工具重新生成。

整理不改变电气设计，也不提高任何验收状态。计算、参考模型及DRC通过不等于硬件、固件、PCB或实物验收；未决项以各模块当前检查为准。PCB布局布线由人类完成。
''')

put('BMS模块/README.md','''# 9S4P BMS

当前工程：[BMS-9S4P.eprj3](BMS-9S4P.eprj3)，必须与同目录`sch/`配套保留。交付仅使用EPRJ3，不新增其它CAD导出格式。

206个板上元件、51种归一化采购MPN（52个原始MPN字符串）；含2个板外CAD载体，共208个位号、700针。局部阻容连线已调整，285个网标、1246条活动LINE、331个导线组；网络及采购身份保持一致。

当前同字节副本重开PASS，正式根路径仍为`HOLD_OPEN_DOCUMENT_TIMEOUT`。DRC为0致命、0错误、12组警告，明细待用户导出核查。可按约束进行可回退初摆；冻结布线和投板保持HOLD。几何、模型与实物验收范围见当前检查，不将模型结果当作物理保证。

| 内容 | 入口 |
|---|---|
| 当前要求与待办 | [项目要求](项目要求.md) · [TODO](TODO.md) |
| 当前状态与证据 | [设计状态](检查/当前设计状态.md) · [DRC修复结论](检查/DRC修复结论.md) · [局部连线](检查/局部连线/结论.md) · [当前发布](检查/Layout验证/当前发布.json) |
| 原理图设计输入 | [原理图准备](原理图准备/) · [BOM](原理图准备/BOM.csv) |
| Layout与接口约束 | [Layout交接](原理图准备/Layout交接.md) |
| 当前计算与模型 | [结果索引](仿真/结果/README.md) · [源码](仿真/源码/) · [输入](仿真/输入/) |
| 历史恢复 | [历史与调试.zip](归档/历史与调试.zip) · [本轮整理记录](../归档/目录整理.json) |

BMS仅经CAN上传温度，独立冷却板自行控制；板上无风扇供电、驱动或测速。5V使用海凌科URB4805YMD-10WR3，输入取保险下游TAP_B9。电池净充电16A为上界，资格不足时CCL=0；30W太阳能由独立MPPT接母线，超额回馈由超容双向DC/DC承接，整车回馈能力仍待资格验证。PSU故障分断、预充实体关断等开放项继续保留。

目录按[电力工程约定](../README.md)维护；当前证据和可复现模型保留，旧风扇/旧方案测试和过程副本收入历史ZIP。PCB布局布线与固件由人类完成。
''')

p=ROOT/'BMS模块/项目要求.md';text=p.read_text(encoding='utf-8-sig')
marker='| U24 | 整理整个电力目录'
if marker not in text:
    line='| U24 | 整理整个电力目录，减少冗余报告和临时测试，保持各板目录简洁统一。 | 用户2026-10-16最新要求。保留一份当前工程、当前必要证据和可复现模型；旧方案与过程材料归入统一ZIP并做SHA256读回后清理。BMS根EPRJ3及sch位置继续遵守U22，不因整理迁移；目录整理不改变电气验收状态。 |\n\n'
    text=text.replace('## 3. 既有工程输入与范围',line+'## 3. 既有工程输入与范围')
put('BMS模块/项目要求.md',text)

p=ROOT/'BMS模块/仿真/结果/README.md';text=p.read_text(encoding='utf-8-sig')
text=text.replace('此目录含当前纸面/静态证据和旧方案结果。','此目录保留当前模型与必要对比证据。')
text=re.sub(r'旧风扇直供/折返、SGM、URB启动[\s\S]*$', '已撤回的风扇供电/折返、独立HLK_CTRL候选和电阻/PACK_LD归并方案已收入[历史与调试.zip](../../归档/历史与调试.zip)，成员映射见[整理记录](../../../归档/目录整理.json)。HLK实现模型及其旧命名依赖因当前脚本仍引用而保留；其中风扇敏感性场景不代表当前BMS存在板载风扇。',text)
put('BMS模块/仿真/结果/README.md',text)

p=ROOT/'元件封装库/README.md';text=p.read_text(encoding='utf-8-sig')
text=re.sub(r'本轮授权修复与复查见[^\n]+', '现行采用型号、用量与采购差异以[来源配置](来源配置.json)、[同步证据](检查/同步证据.json)和索引为准。跨板共用候选不等于已经替换或完成电气验证。',text)
text=re.sub(r'2026-10-16：BMS当前原生入口[^\n]+', 'BMS当前原生入口为模块根EPRJ3及配套sch，状态以[BMS当前设计状态](../BMS模块/检查/当前设计状态.md)为准；局部连线调整后为208个位号、700针、285个网标。当前副本重开PASS，正式根路径HOLD，DRC为0致命、0错误、12组警告待审，Layout正式放行仍HOLD。库一致性校验不替代电气或实物验收。',text)
put('元件封装库/README.md',re.sub(r'\n{3,}','\n\n',text))

for board in ['FOC驱动模块','稳压模块']:
    p=ROOT/board/'README.md';text=p.read_text(encoding='utf-8-sig')
    text += '\n\n## 目录维护\n\n目录按[统一约定](../README.md)维护。当前工程、必要检查证据和可复现模型原位保留，旧方案及过程材料收入`归档/历史与调试.zip`；恢复映射与SHA256读回见[本轮整理记录](../归档/目录整理.json)。整理不改变设计或验收状态。\n'
    if board=='稳压模块':
        text=text.replace('V1.5 LICAP350F 输入及既有结果仍为历史模型。','旧V1.5 LICAP350F输入、源码及结果已收入历史ZIP，不作为当前模型入口。')
        text=text.replace('原件另按整理记录保留在根目录的可恢复材料区。','原件保留或清理情况以本轮整理记录为准。')
    put(board+'/README.md',re.sub(r'\n{3,}','\n\n',text))

# 仅改向本轮归档的现存Markdown链接，不改计算源码或机器证据里的历史路径。
changes=[]
for base in ['BMS模块','FOC驱动模块','稳压模块','元件封装库']:
    for p in (ROOT/base).rglob('*.md'):
        rel=p.relative_to(ROOT).as_posix()
        if rel in removed or '/仿真/输入/' in rel or any(x in p.parts for x in ['归档','.git','参考资料','厂商资料','共享仓库','原生片段']):continue
        text=p.read_text(encoding='utf-8-sig')
        def replace(m):
            raw=m.group(2);target=unquote(raw.strip('<>')).split('#')[0]
            if not target or '://' in target or target.startswith('#'):return m.group(0)
            q=Path(target) if Path(target).is_absolute() else p.parent/target
            try: key=q.resolve().relative_to(ROOT).as_posix()
            except (ValueError,OSError):return m.group(0)
            if key in removed: row=removed[key]
            elif not q.exists():
                candidates=[v for k,v in removed.items() if k.startswith(key.rstrip('/')+'/')]
                if not candidates:return m.group(0)
                row=candidates[0]
            else:return m.group(0)
            dest=os.path.relpath(ROOT/row['archive'],p.parent).replace('\\','/')
            changes.append({'document':rel,'old':raw,'archive':row['archive'],'member':row['member']})
            return '['+m.group(1)+']('+dest+')（历史成员：`'+row['member']+'`）'
        updated=re.sub(r'\[([^\]\n]+)\]\(([^)\n]+)\)',replace,text)
        if base=='BMS模块':
            updated=updated.replace('](../eda工程文件/BMS-9S4P.eprj3)','](../BMS-9S4P.eprj3)')
            updated=updated.replace('](../eda工程文件/sch/Schematic1/P1.esch2)','](../sch/Schematic1/P1.esch2)')
            if rel=='BMS模块/原理图准备/功率与预充设计.md':
                updated=updated.replace('](../稳压模块/参考资料/','](../../稳压模块/参考资料/')
        if updated!=text:
            # 修改前正文也写入本轮归档，并读回校验。
            dest=base+'/归档/历史与调试.zip' if base!='元件封装库' else '归档/工程过程.zip'
            member='20261016_目录整理/链接更新前/'+rel;raw=p.read_bytes()
            with zipfile.ZipFile(ROOT/dest,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                z.writestr(member,raw)
            with zipfile.ZipFile(ROOT/dest) as z:
                if hashlib.sha256(z.read(member)).digest()!=hashlib.sha256(raw).digest():raise RuntimeError('文档归档失败')
            document_backups.append({'path':rel,'archive':dest,'member':member,'sha256':hashlib.sha256(raw).hexdigest(),'readback':True})
            p.write_text(updated,encoding='utf-8')
receipt['markdown_archive_links']=changes
receipt['document_backups']=document_backups
for rel,info in receipt['archives'].items(): info['documents_sha256']=file_sha(ROOT/rel)
(ROOT/'归档/目录整理.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('当前入口已更新；历史链接改向',len(changes))
