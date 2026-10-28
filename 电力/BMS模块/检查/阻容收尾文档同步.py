"""同步最终阻容设计说明；不执行电气测试。"""
from pathlib import Path
B=Path(__file__).resolve().parents[1]
S=B/'原理图准备'; C=B/'检查'
for p in S.glob('*.md'):
    t=p.read_text(encoding='utf-8-sig')
    t=t.replace('54.9kΩ/100kΩ及33.2kΩ/10kΩ参考复算','60.4kΩ/100kΩ及33.2kΩ/10kΩ参考复算')
    t=t.replace('约10.25°C冷停、59.95°C热停为连接源标称值','含200kΩ传感负载后的模型标称冷停约10.46°C、热停约59.18°C')
    t=t.replace('物理元件未变，未编程或测试。','本轮已将温度参考改为60.4kΩ/33.2kΩ并保存原生，计算见[阻值归并计算](../检查/均衡阻值归并计算.md)；未编程或测试。')
    t=t.replace('原理图使用通用R/C封装载体的项目只定义阻值、电容值和初始封装，采购前按额定电压、功率、精度和DC偏压选定真实料号。','本轮132个均衡阻容位号已选定37种真实MPN并绑定原生器件，见检查/通用阻容落实记录.json；采购价格、MLCC有效容量、实际焊盘和降额仍待人工核定。')
    p.write_text(t,encoding='utf-8')
p=C/'需求覆盖与待验收.md'; t=p.read_text(encoding='utf-8-sig')
t=t.replace('通用R/C载体规定了值和初始封装，真实采购料还需按耐压、功率、精度和DC偏压选定。封装关联回读不等于pad映射验收。','132个均衡阻容位号已选定37种真实MPN并保存原生；MLCC有效容量、功率降额、报价和实际焊盘仍需人工核定。封装关联回读不等于pad映射验收。')
p.write_text(t,encoding='utf-8')
p=B/'README.md'; t=p.read_text(encoding='utf-8-sig')
t=t.replace('物理元件未改，参数尚未编程或实物验收。','温度参考已改为60.4kΩ/33.2kΩ，含传感节点200kΩ负载的模型冷停约10.46°C、热停约59.18°C；参数尚未编程或实物验收。')
if '## 通用阻容收尾' not in t:
    t+='\n## 通用阻容收尾（2026-10-13）\n\n132个均衡阻容位号已选定37种真实MPN并同步当前BOM、原生工程和官方导出。第2–4组缓冲OV参考归并为100kΩ，标称差分门限2.50V；第1组保留2.55V。反馈主上臂与下臂均为0.1%，没有新增元件。详见[落实记录](检查/通用阻容落实记录.json)及[归并计算](检查/均衡阻值归并计算.md)。完整报价、MLCC有效容量、焊盘与电气验收仍待完成。\n'
p.write_text(t,encoding='utf-8')
p=C/'LM51772低压边界.md'; t=p.read_text(encoding='utf-8-sig')
lines=t.splitlines()
for i,l in enumerate(lines):
    if l.startswith('- **取样共模仅完成静态角落计算。**'):
        lines[i]='- **取样共模仅完成静态角落计算。** 主上臂523kΩ和下臂100kΩ均采用0.1%，并联支路2.43MΩ为1%。计入1V基准±3%、电阻容差和FB最大±60nA后，最低BUF约5.1038V；暂按52mV反向shunt压降扣减，较低ISNS脚约5.0518V，距5V约51.8mV。负向52mV、档位MOS漏电、温漂和动态纹波尚未保证，此结果不能宣称全工况共模满足；见[阻值归并计算](均衡阻值归并计算.md)。'
    elif l.startswith('| 外部反馈 |'):
        lines[i]='| 外部反馈 | 在shunt后BUF取样。523kΩ/0.1%上臂、100kΩ/0.1%下臂；目标阶段并联2.43MΩ/1%。标称低/高5.3037/6.23V。含基准、电阻及FB最大±60nA的静态角落：低5.1038–5.5046V，高6.0016–6.4591V；不得再次加减漏电误差。动态、温漂和档位MOS漏电待核。 |'
    elif l.startswith('| ISNS共模角落 |'):
        lines[i]='| ISNS共模角落 | 当前静态算例以最低BUF5.1038V减去假设52mV，较低ISNS脚约5.0518V、余量51.8mV。未含动态/温漂与开关漏电，且负向52mV无保证，不作为共模验收。 |'
    elif l.startswith('仅以Vsource=3.05V'):
        lines[i]='仅以Vsource=3.05V、VBUF_high约6.4591V（电阻/基准及FB漏电静态高角落）、Lmin=17.6µH、fs=195.827kHz（仅RRT误差）、ηch=85%估算，ΔIL约0.467A，buffer充电电流约1.71A。该值未包含控制器振荡器容差、电感随电流/温度变化、效率变化和实际波形，属于设计计算，不能证明全周期净电流。'
    elif l.startswith('- 软件默认均衡温度窗口'):
        lines[i]=l.replace('≤10°C另有硬件禁止，10–15°C软件保持OFF','浮地硬件另有模型标称约10.46°C冷停；10–15°C软件保持OFF，实际硬件温度误差待核')
p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
p=B/'TODO.md'; t=p.read_text(encoding='utf-8-sig')
t=t.replace('- [ ] 同步当前连接源、BOM、原生原理图和官方导出，并记录本轮完成范围。','- [x] 同步当前连接源、BOM、原生原理图和官方导出，并记录本轮完成范围；官方回读43页/594个part符号，BOM653行。')
t=t.replace('热保护标称停止温度待计算报告核定','热保护模型标称停止温度约59.18°C')
t=t.replace('- [ ] 完成33.2kΩ热保护阈值计算，并按计算代理最终报告更新温度策略。','- [x] 完成33.2kΩ热保护阈值计算；含200kΩ分压负载的模型标称热停约59.18°C，NTC/比较器及安装误差仍待闭合。')
t=t.replace('完成通用阻容真实采购型号建议：','落实通用阻容真实采购型号：')
p.write_text(t,encoding='utf-8')
print('final passive documentation synchronized')
