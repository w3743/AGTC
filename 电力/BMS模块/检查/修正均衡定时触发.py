"""Correct TI timer trigger nets after reading the manufacturer functional diagram."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import zipfile

module = Path(__file__).resolve().parents[1]
source = module / '原理图准备/共享电感控制保护连接.json'
rows = json.loads(source.read_text(encoding='utf-8-sig'))
timer = next(r for r in rows if r['ref'] == 'U_BAL_TIMER')
expected = {'2':'BAL_CS_N', '10':'BAL_RUN', '5':'BAL_BLANK',
            '12':'BAL_NOT_BLANK', '13':'BAL_WD_OK',
            '6':'BAL_BLANK_CX', '7':'BAL_BLANK_RX',
            '14':'BAL_WD_CX', '15':'BAL_WD_RX'}
if any(timer['pins'].get(k) != v for k,v in expected.items()):
    raise RuntimeError('Timer source differs from the inspected snapshot')
paths = [source, module/'原理图准备/BOM.csv', module/'原理图准备/连接表.csv',
         module/'检查/共享电感缓冲电容证据.md', module/'检查/共享电感控制实际减件方案.md']
old_sha = hashlib.sha256(source.read_bytes()).hexdigest().upper()
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
with zipfile.ZipFile(module/'归档/历史与调试.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
    for p in paths:
        raw = p.read_bytes()
        entry = '定时触发修正前_' + stamp + '/' + p.relative_to(module).as_posix()
        archive.writestr(entry, raw)
        if hashlib.sha256(archive.read(entry)).digest() != hashlib.sha256(raw).digest():
            raise RuntimeError('Archive SHA256 mismatch')

timer['pins']['2'] = 'BAL_RUN'
timer['pins']['10'] = 'BAL_CS_N'
timer['note'] = ('TI CD74HCT123M原厂SCHS142F第2页功能图：CH1 inputs1/2/3，Q13 /Q4，Cx14/RxCx15；'
                 'pin2=BAL_RUN，每50ms来自健康任务的上升沿续约约450ms看门狗，Q13=BAL_WD_OK。'
                 'CH2 inputs9/10/11，Q5 /Q12，Cx6/RxCx7；pin10=BAL_CS_N，锁存边沿触发约30.6ms换接空窗，'
                 'Q5=BAL_BLANK，/Q12=BAL_NOT_BLANK。RC、输出、PG复位连接不变。不能将常见其他厂商74HC123的通道号套入本TI型号；'
                 '禁止DMA/free-running心跳，均衡禁用时OE=0。原厂 https://www.ti.com.cn/cn/lit/gpn/cd74hc123 。'
                 '本次为引脚图驱动的源修正，未运行ERC、仿真或实物测试。')
source.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
new_sha = hashlib.sha256(source.read_bytes()).hexdigest().upper()
for p in paths[3:]:
    s = re.sub(r'\r+\n', '\n', p.read_text(encoding='utf-8-sig'))
    p.write_text(s.replace(old_sha,new_sha), encoding='utf-8')
report = {
    'date':'2026-10-14', 'ref':'U_BAL_TIMER',
    'manufacturerSource':'https://www.ti.com.cn/cn/lit/gpn/cd74hc123',
    'page':2, 'image':'检查/原厂图核对/CD74HCT123-pinout.png',
    'before':{'2':'BAL_CS_N','10':'BAL_RUN'},
    'after':{'2':'BAL_RUN','10':'BAL_CS_N'},
    'unchanged':'RC, Q,/Q, PG复位、器件数量和型号不变',
    'purpose':'看门狗由健康任务RUN续约；每个CS上升沿先开启换接空窗，随后延迟锁存595',
    'sourceSha256':new_sha, 'archiveReadback':'SHA256 MATCH',
    'status':'连接源已修正，原生待同步；非测试或电气验收'
}
out = module/'检查/均衡定时触发修正.json'
if out.exists():
    raise RuntimeError('Unexpected existing correction report; preserve it')
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('U_BAL_TIMER pin2 RUN / pin10 CS_N corrected; quantity/type delta 0; archive SHA256 MATCH')
