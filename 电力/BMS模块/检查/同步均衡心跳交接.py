from pathlib import Path
import datetime
import hashlib
import re
import zipfile

module = Path(__file__).resolve().parents[1]
p = module/'原理图准备/编程交接.md'
raw = p.read_bytes()
with zipfile.ZipFile(module/'归档/历史与调试.zip','a',compression=zipfile.ZIP_DEFLATED) as archive:
    entry = '心跳交接修正前_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '/原理图准备/编程交接.md'
    archive.writestr(entry,raw)
    if hashlib.sha256(archive.read(entry)).digest() != hashlib.sha256(raw).digest():
        raise RuntimeError('Archive mismatch')
s = re.sub(r'\r+\n','\n',raw.decode('utf-8-sig'))
s = re.sub(r'^- \*\*硬件刷新心跳：\*\*.*$',
    '- **硬件刷新心跳：**健康均衡任务每50ms左右使`BAL_RUN`产生上升沿，续约约450ms硬件看门狗。RUN不触发换接空窗，不使用DMA或无条件中断续命；均衡禁用时OE=0，可停止RUN。`BAL_CS_N`只在写入新的地址/方向/阶段请求时产生锁存边沿，每一边沿触发独立空窗。RUN静态高或低不能续约，该心跳不替代BQ主保护链。', s, flags=re.M)
s = re.sub(r'^- \*\*每次换接：\*\*.*$',
    '- **每次换接：**严格按[共享电感编程接口的正常停止与换组](共享电感编程接口.md)执行：先I²C关闭转换并回读、等待持续零流；再撤销OE、写新址及bit5=0，完成第一段完整空窗；重新建立硬件许可后写bit5=1，完成第二段独立空窗，最后才设置CONV_EN=1。任一超时进入故障停车，不继续换接。', s, flags=re.M)
s = s.replace('在`CONV_OFF`下禁用不用的VDET/同步功能', '在D0.CONV_EN=0并回读确认后禁用不用的VDET/同步功能')
s = s.replace('Hall电流正方向为源电芯→转换器', 'Hall位于缓冲端，正方向为转换器→缓冲，负方向为缓冲→转换器')
p.write_text(s,encoding='utf-8')
print('Programming handoff: RUN watchdog, CS blank, two-stage changeover and buffer Hall synchronized; archived SHA256 MATCH')
