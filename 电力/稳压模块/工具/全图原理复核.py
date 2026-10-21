"""按当前ENET逐脚核对设计合同，并记录完整功能审查；不修改EDA源/PCB。"""
from pathlib import Path
import csv
import hashlib
import json
import math
import zipfile

M = Path(__file__).resolve().parents[1]
O = M / '检查'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

plan = load(M/'原理图准备/design-plan.json')
material_proof_path = O/'原理整改/PEH226NL80V_变更证明.json'
material_proof = load(material_proof_path)
native_page = next((M/'eda工程文件/sc-regulator').rglob('*.esch2'))
assert sha(native_page) == material_proof['nativeProof']['afterSha256']
assert sha(M/'原理图准备/BOM.csv') == material_proof['bomProof']['afterSha256']
assert sha(M/'原理图准备/design-plan.json') == material_proof['planProof']['afterSha256']
# Reproduce the byte-level claim rather than trusting the stored PASS flags.
archive = O/'原理整改/PEH226NL80V_修改前源.zip'
assert sha(archive) == material_proof['backup']['sha256']
with zipfile.ZipFile(archive) as z:
    members = [n for n in z.namelist() if n.endswith(native_page.name)]
    assert len(members) == 1
    baseline = z.read(members[0])
assert hashlib.sha256(baseline).hexdigest() == material_proof['nativeProof']['beforeSha256']
old_lines, new_lines = baseline.splitlines(keepends=True), native_page.read_bytes().splitlines(keepends=True)
assert len(old_lines) == len(new_lines)
allowed = {(r['parentId'], r['key']) for r in material_proof['changedNativeAttributes']}
changed_count = 0
for old, new in zip(old_lines, new_lines):
    if old == new: continue
    old_header, old_body = old.decode('utf-8').split('||', 1)
    new_header, new_body = new.decode('utf-8').split('||', 1)
    assert old_header == new_header and json.loads(new_header)['type'] == 'ATTR'
    a, b = json.loads(old_body.rstrip('\r\n|')), json.loads(new_body.rstrip('\r\n|'))
    assert (a['parentId'], a['key']) in allowed
    assert {k:v for k,v in a.items() if k != 'value'} == {k:v for k,v in b.items() if k != 'value'}
    changed_count += 1
assert changed_count == 48
enetpath = M/'检查/当前官方网表.enet'
enet = load(enetpath)
actual = {c['props']['Designator']: c for c in enet['components'].values()}
expected = {c['ref']: c for c in plan['components']+plan.get('systemBoundaries', [])}
assert set(expected) == set(actual), '不能漏检位号'
rows = []
for ref, c in actual.items():
    pins = expected[ref]['pins']
    assert set(pins) == set(c['pinInfoMap']), (ref, '引脚集合不一致')
    for number, p in c['pinInfoMap'].items():
        target = pins[number] or ''
        rows.append({'ref': ref, 'pin': number, 'name': p['name'],
                     'expectedNet': target, 'actualNet': p['net'],
                     'status': 'PASS' if target == p['net'] else 'FAIL'})
assert all(r['status']=='PASS' for r in rows)

# 每个实际器件均归入且只归入一块功能；功能检查和合同一致性分别记录。
def domain(ref):
    g = expected[ref].get('visualGroup', '')
    if ref == 'SYS_BAT_BUS': return '系统电源边界'
    if ref.startswith('SC') and ref[2:].isdigit(): return '固定超容与串联抽头'
    if g.startswith('cell'): return '逐节采样及均衡'
    return {'power':'功率桥/隔离/电感/电容/熔断', 'driver':'DRV8161及栅极网络',
            'aux12':'12V辅助电源', 'scov':'总压硬件比较', 'dump':'泄放功率及驱动',
            'core1':'核心板排针', 'core2':'核心板排针', 'logic_supply':'5V/3.3V电源边界',
            'busadc':'BUS采样', 'testfb':'抽头测试回读', 'can':'CAN通信及终端',
            'taps':'抽头接口', 'ntc_SC':'三路温度采样', 'ntc_MAIN':'三路温度采样',
            'ntc_DUMP':'三路温度采样'}.get(g, '其他已逐件检查')
groups = {}
for ref in actual: groups.setdefault(domain(ref), []).append(ref)

def item(id, state, subject, finding, action):
    return dict(id=id,status=state,subject=subject,finding=finding,action=action)
findings = [
 item('SC01','FAIL','容量/保护实现',
      '现行原生仍是8S1P、8只固定LICAP。8S2P/16只只是候选；800W双向1s模型有附加条件，不能写成已落图或1kW已通过。',
      '保持固定型号和16只上限；把逐组测量、双向限流和失效隔离一起闭合后再落图，不能只加电容。'),
 item('SC02','FAIL','逐节差分测量',
      'T1..7为68k/10k并联47k TEST支路，TEST为低时比例9.24681；T8比例7.8。累积抽头相减会累加电阻比、ADC、VREF、漏电及shunt参考误差；既有预算不满足逐组保护。',
      '停止把此链作为已证明单体硬保护；优先复用BMS的BQ76952候选，仍须解决0V首次充电、输入诊断和硬关断。'),
 item('SC03','PASS_WITH_LIMITS','BW6101脚序/真实阈值',
      '8路均为VDD=组+、GND/SEL=组-、LED悬空、IOUT经120Ω回组-。原厂SEL低为2.45V(25°C范围2.42..2.48)，不是2.65V；外部电阻把泄流限制到约20mA量级，不能替代几十A充电关断。',
      '按2.45V被动均衡建模，确认软充电上限/静态损耗；不能用原厂裸芯700mA数字替代本支路电流。'),
 item('PW01','FAIL','故障隔离',
      'Q_BUS_P源=BUS_BR、漏=BUS_EXT，其体二极管允许BUS_EXT向BUS_BR供电。QH D-S短路时MCU关PWM/关P管不能阻断向超容充电；80A保险不保证在中等持续过流下及时清除。',
      '比较复用双N管的双向阻断与背靠背P管；须覆盖默认失电、器件短路、0V充电及20uH电感断流能量。'),
 item('PW02','OPEN','P管驱动/损耗',
      '10V齐纳低端不能保证VGS达到IXTH76P10T的-10V RDS测试条件；47k/10k静态栅极网络未证明快速故障隔离。R_QP_COL仅0.1W，较高BUS边界下功耗超过额定。',
      '若保留P管，重算全温VGS、上拉关断时间、串阻功率、热态损耗及SOA；优先削减高损耗P管方案的系统成本。'),
 item('PW03','OPEN','主桥/磁件/熔断',
      'QH D=BUS_BR/S=SW，QL D=SW/S=GND，2只10uH串联后接SC_FUSED，0.5mΩ置SC负端，连接方向一致。12uH热态80A最小值、快限流延迟和外部AMG80的DC故障清除未证明。',
      '以实测L-I-T和端到端关断延迟限定能力；不能把0.8V VDS OCP视为精确80A限流。'),
 item('PW04','PASS_RATING_MISMATCH_FIXED_DYNAMIC_OPEN','TVS与被保护器件耐压',
      '当前Boarden SMCJ43A保留在BUS_EXT，六只bulk已改为PEH226NL3900QE4、900µF/80V。原63V额定低于SMCJ43A表列最大钳位69.4V@21.7A的问题已修正；80V与此规定脉冲数据有10.6V名义间隔，但并非实际瞬态保证。43V只是VRWM。现有SMCJ40A钳位64.5V@23.3A、VRWM40V，对37.8V正常母线仅2.2V余量，本轮未切换。实际脉冲电流、线束寄生与重复能量尚未闭合。',
      '按实测正常最高电压和最坏断流能量复核TVS及电容的联合窗口；不能只按单个VC值定型，也不把69.4V写成每次必达。'),
 item('PW05','PASS_WITH_LIMITS','母线电容位置与纹波',
      '已实施6×900uF/80V，实际在BUS_EXT；BUS_BR只有3×2.2uF薄膜、1uF和100nF。六bulk在125°C、≥5kHz自然对流条件下额定纹波理想总和44.4A，仅为筛选；关断P管后不能继续把5.4mF计在桥侧。',
      '按真实节点进行P管RMS、电容均流、局部换流和关断能量分析，实测ESL/尖峰/温升。'),
 item('GD01','PASS_WITH_LIMITS','DRV8161全部20脚',
      '引脚与TI DGS定义相符。3.3k DT为双输入带防直通，典型370ns；ID1/2均5.6k为典型512/768mA；30k增益20；27k VDS典型0.8V。BOOT接SW、VDRAIN接BUS_BR、SP/SL=GND、SN=SC_MINUS，CSAREF接核心板3.3V且去耦。',
      '保持低态启动；CSA输入±1.94mV偏置等效±3.88A仅25°C项，快保护必须校准并覆盖温漂/增益/延迟。'),
 item('GD02','OPEN','共享nFAULT许可电平',
      'PB5经1k推高共享故障节点，10k下拉、TLV OUTB开漏拉低。名义高约3.0V，但MCU保证VOH/3.0V低供电角落与输入阈值的裕量尚未形成完整证明。MCU挂高只剩总压保护，缺逐组硬许可。',
      '为实际控制输出建立最小VOH/最大灌电流预算并落硬件逐组许可，避免以典型3.3V计算作为保证。'),
 item('OV01','PASS_WITH_LIMITS','TLV6700全部6脚',
      'VDD=12V_DRV、GND/INA+=SC_MINUS、INB=2×255k/10k总压分压、OUTB接故障线、OUTA悬空，逻辑正确。总压典型20.8V，容差边界约20.57..21.05V，不保护失衡单体；12V失电不保证有效输出。',
      '保留其为独立总压后备；不能代替逐组OV/UV、失电隔离和启动资格。'),
 item('DU01','FAIL','dump保护独立性',
      'UCC27511的IN-=GND，IN+=PB7经100Ω，输入及门极下拉齐全；12V驱动掉电/MCU停止时dump停用。当前稳压板没有独立硬件BUS过压dump，板外1.8Ω电阻也没有实际脉冲/热资格。',
      '按两块FOC现有独立dump与中央dump的整车能量分工删减重复成本，仍须证明任一要求场景有有效能量出口。'),
 item('AD01','FAIL','BUS ADC过压/核心板失电',
      '当前150k/10k比例16；37.8V为2.3625V。钳位若到69.4V则ADC为4.3375V，超过3.3V模拟电源允许范围。live BUS/SC而核心板断电还可通过采样输入反灌。',
      '重算分压及抗失电开关/限注入，所有正常和瞬态边界均需满足；候选TMUX1511离电信号上限3.6V不能承接4.3375V。'),
 item('AX01','PASS_WITH_LIMITS','12V模块身份、脚位和启动边界',
      'U_AUX1的HiLink厂家与URB4812YMD-10WR3型号由本地HiLink原厂资料直接匹配（参考资料/海凌科_URAB_YMD-10WR3_V1.7.pdf，SHA256 1c55e95b4c6457f75984519c61ba2c544b15f9e65016e8ebf4cc286b545ff49d）。原厂限值：输入18..75V、输出12V/833mA、最大容性负载680uF；1=负输入、2=正输入、3=正输出、4=NC、5=输出回路、6=CTRL，CTRL悬空为开启。当前BUS_EXT、12V_DRV、GND_SYS与1/2/3/5连线相符，4 NC及6悬空配置相符。要求最小负载为额定输出的3%；390Ω/2W在12V下名义0.369W/30.77mA，即约3.69%，可作预载候选。',
      '不再把厂家身份列为歧义，也不借用MORNSUN参数。仍需实测BUS_EXT最低输入启动、实物标签与装配方向、总输出容性负载不超680uF、输出预载温升及整机热稳态；这些未测前不提升全温/量产资格。'),
 item('IO01','PASS_WITH_LIMITS','核心板全部48脚/通信/温度',
      '48脚全部核对合同；PA8/PB13 TIM1互补输出、PB1/PA7电流、PB5许可、PB6 P管、PB7 dump，CAN PA11/12与TCAN332、3路NTC、TEST回读均一致。PB2/VREF脚未占用。PB6与PA9板载跳桥、BOOT0配置、USB反灌仍要求实板读回。',
      '强制按WeAct实际修订确认SB3/SB6和外部5V/USB路径；CAN终端仅在线路两端装配。NTC开短路状态须进入保护，不用温度正常值掩盖故障。'),
 item('IO02','OPEN','0V正常充电',
      '超级电容从0V正常可充，不需恢复激活；目前硬件没有证明闭环限流初充、低压测量/驱动建立、自举最小脉宽及硬许可时序完整。原8S1P与候选BQ从SC供电均不能靠文字认定闭合。',
      '复用双向DC/DC进行受控初充；独立限流电源只用于台架，不将其写为用户日常维护要求，也不允许直接并接9S BUS。'),
]

calc = {
 'busDivider':{'ratio':16,'at37_8V':37.8/16,'at69_4V':69.4/16},
 'tap1to7TestLowRatio':1+68000/(1/(1/10000+1/47000)),
 'tap8Ratio':7.8,
 'balanceResistor':{'nominalOhm':120,'upperCurrentAt2_48V_A':2.48/120,'upperPowerAt2_48V_W':2.48**2/120},
 'shuntAt80A':{'dropV':80*.0005,'powerW':80**2*.0005},
 'csaNominal':{'zeroV':1.65,'slopeVperA':.01,'atPositive80A':2.45,'atNegative80A':.85,'offsetErrorAt25C_A':.00194/.0005},
 'auxPreloadW':12**2/390,
 'auxPreloadCurrentA':12/390,
 'auxPreloadPercentRatedOutput':(12**2/390)/10*100,
 'dumpRminOhm':1.8*.99,
 'dumpAt42V_W':42**2/(1.8*.99),
 'dumpAt45_59V_W':45.59**2/(1.8*.99),
 'pGateCollectorResistor':{'at37_8VAssume10Vz_W':(37.8-10)**2/10000,
                           'at45_59VAssume10Vz_W':(45.59-10)**2/10000},
 'bulkBusExt_F':6*900e-6,
 'bridgeLocalNominal_F':(3*2.2+1+.1)*1e-6,
 'inductorEnergyAt80A20uH_J':.5*20e-6*80**2,
 'sc2x100uFMin160uFIdealEnergyRiseFrom19_6V':math.sqrt(19.6**2+(20e-6*80**2)/160e-6),
}

# Exact BUS capacitor/TVS reuse screen, tied to the last generated central index.
library_index_path = M.parent/'元件封装库/索引/元件总表.json'
library_index = load(library_index_path)
library_index_sha = sha(library_index_path)
source_freshness = {}
for source in library_index.get('sources', []):
    source_path = M.parent/source['path']
    current_sha = sha(source_path) if source_path.exists() else None
    source_freshness[source['module']] = {
        'path': source['path'], 'indexedSha256': source['sha256'],
        'currentSha256': current_sha, 'match': current_sha == source['sha256']
    }

def indexed_rows(mpn_fragment):
    return [row for row in library_index.get('instances', []) if mpn_fragment.casefold() in str(row.get('mpn') or '').casefold()]

def indexed_usage(mpn_fragment):
    found = indexed_rows(mpn_fragment)
    per_module = {}
    for row in found:
        per_module.setdefault(row['module'], []).append(row['ref'])
    vehicle_count = sum(len(refs) * (2 if module == 'FOC驱动模块' else 1)
                        for module, refs in per_module.items())
    stale_modules = sorted(module for module in per_module if not source_freshness.get(module, {}).get('match', False))
    return {
        'indexInstanceCount': len(found), 'perModuleDesignators': per_module,
        'vehicleCountEstimate': vehicle_count, 'staleSourceModules': stale_modules,
        'vehicleCountStatus': 'PROVISIONAL_INDEX_SOURCE_STALE' if stale_modules else 'INDEX_SOURCE_MATCH'
    }

bus_power_screen = {
    'asOf': '2026-10-21',
    'centralIndex': {
        'path': library_index_path.relative_to(M.parent).as_posix(),
        'sha256': library_index_sha, 'sourceFreshness': source_freshness,
        'vehicleQuantityBasis': 'FOC驱动模块按2块，其余已索引模块按1份；任一来源SHA不匹配时其用量标为临时。'
    },
    'bulkCapacitor': {
        'current': {
            'manufacturer': 'KEMET', 'mpn': 'PEH226NL3900QE4',
            'referenceCount': 6, 'designators': [f'C_BUS_BULK{i}' for i in range(1, 7)],
            'ratedVoltageV': 80, 'capacitanceEachUf': 900, 'totalCapacitanceUf': 5400,
            'rippleRatedNaturalConvection125CAtGe5kHz_A_each': 7.4,
            'rippleRatedNaturalConvection125CAtGe5kHz_A_total': 44.4,
            'bomFootprintStatus': 'original_ML_CAD_carrier_retained; exact_NL_footprint_qualification_pending',
            'implementationStatus': 'IMPLEMENTED_NATIVE_ATTR_AND_BOM; PRICE_NOT_QUOTED'
        },
        'sameSeriesCandidate': {
            'status': 'IMPLEMENTED; legacy schema key retained for compatibility',
            'manufacturer': 'KEMET/YAGEO', 'mpn': 'PEH226NL3900QE4',
            'referenceCountForSameNominalCapacitance': 6,
            'ratedVoltageV': 80, 'capacitanceEachUf': 900, 'totalCapacitanceUf': 5400,
            'ratedRippleNaturalConvection125CAtGe5kHz_A_each': 7.4,
            'ratedRippleNaturalConvection125CAtGe5kHz_A_total': 44.4,
            'maxRippleNaturalConvection125CAtGe5kHz_A_each_notContinuousRating': 9.4,
            'rippleAt150CWithHeatSink_A_each': 5.7,
            'rippleAt150CWithHeatSink_A_total': 34.2,
            'nominalCase': 'size L; 20.2mm x 43.7mm; 15.24mm lead spacing',
            'compatibility': '几何尺寸同级但真实封装仍待导入/核对；2026-04-28数据表注明L规格新设计采用top-weld crown，目录件可能仍为side-weld，不作drop-in声明。',
            'deltaFromCurrent': {
                'comparisonBasis': 'previous 63V ML baseline, not the present 80V selection',
                'ratedVoltageIncreaseV': 17, 'capacitanceChangeUf': 0,
                'ratedNaturalConvection125CAtGe5kHzRippleChange_A_total': -4.2,
                'ratedRippleChangePercent': -8.641975308641975,
                '40ArmsHeadroomAtThatOneRatedCondition_A': 4.4,
                'fullTemperature40ArmsAcceptance': 'NOT_ESTABLISHED'
            }
        },
        'existing100VCentralPart': {
            'manufacturer': 'KEMET', 'mpn': 'PEH200PJ5100MB2',
            'indexReferencesPerFOCBoard': 3, 'vehicleEstimateWithFOCx2': 6,
            'ratedVoltageV': 100, 'capacitanceEachUf': 10000,
            'caseAndMounting': '50mm x 95mm screw-terminal can',
            'reuseAssessment': '已分配FOC母线电容；体积、螺栓端子与当前径向15.24mm候选焊盘不同，不是稳压板drop-in。'
        },
        'electricalBoundary': '电容承压按VDC + peak VAC不得超过额定值。已选80V器件相对SMCJ43A表列69.4V钳位仅有10.6V名义差值；实际浪涌、走线尖峰和电流必须实测。'
    },
    'busTvs': {
        'normalBusUpperReferenceV': 37.8,
        'options': [
            {'manufacturer':'Boarden','mpn':'SMCJ43A','designators':['稳压模块:D_BUS_TVS'],
             'indexUsage':indexed_usage('SMCJ43A'),'vrwmV':43,'vbrMinMaxV':[47.8,52.8],
             'vcMaxV':69.4,'ippA':21.7,'headroomTo37_8V_V':5.2,
             'datasheet':'Boarden SMCJ-Series, rev.1.1, p.2; local SHA256 cd6f48ce70aa36cc1ec26b499838fcf44face4eed34dcd9e1e45ebb0bf6139dd'},
            {'manufacturer':'Littelfuse','mpn':'SMCJ40A','designators':['FOC驱动模块:DTVS1','水泵驱动模块:D_BUS'],
             'indexUsage':indexed_usage('SMCJ40A'),'vrwmV':40,'vbrMinMaxV':[44.4,49.1],
             'vcMaxV':64.5,'ippA':23.3,'headroomTo37_8V_V':2.2,
             'reuseAssessment':'跨模块已有型号可复用，但37.8V正常上限下仅2.2V至VRWM；需确认全容差/纹波最高电压严格低于40V。'},
            {'manufacturer':'Littelfuse','mpn':'SMCJ45A','designators':['BMS模块:D_TVS'],
             'indexUsage':indexed_usage('SMCJ45A'),'vrwmV':45,'vbrMinMaxV':[50.0,55.3],
             'vcMaxV':72.7,'ippA':20.6,'headroomTo37_8V_V':7.2,
             'reuseAssessment':'有更高VRWM余量，但表列钳位高于SMCJ43A，不能改善bulk电容尖峰余量。'}
        ],
        'ratingBoundary': 'Littelfuse SMCJ为1500W、10/1000µs、0.01%占空比的浪涌指标，按非重复脉冲标定；不构成动力母线重复断流能量或线束寄生的保证。',
        'screenConclusion': '本轮已实施保留SMCJ43A、六只bulk改为PEH226NL3900QE4/80V，数量与总容量不变。未切换SMCJ40A。当前只修复额定不匹配；实际波形、纹波、温升、物理封装及价格仍待验证，不授予释放资格。'
    },
    'primarySources': [
        {'title':'YAGEO/KEMET PEH226 datasheet A4017 (2026-04-28), electrical ratings and dimensions',
         'url':'https://yageogroup.com/content/datasheet/asset/file/KEM_A4017_PEH220_225_226'},
        {'title':'Littelfuse SMCJ series datasheet rev.2 (2025-02-27), electrical table p.2',
         'url':'https://www.littelfuse.com/~/media/electronics/datasheets/tvs_diodes/littelfuse_tvs_diode_smcj_datasheet.pdf.pdf'},
        {'title':'KEMET PEH200 datasheet A4034 (2023-08-08), ratings p.7 and dimensions p.3',
         'url':'https://content.kemet.com/datasheets/KEM_A4034_PEH200.pdf'},
        {'title':'Boarden SMCJ Series rev.1.1 p.2',
         'localPath':'参考资料/Boarden_SMCJ-Series.pdf',
         'sha256':'cd6f48ce70aa36cc1ec26b499838fcf44face4eed34dcd9e1e45ebb0bf6139dd',
         'url':'https://boardensh.com/wp-content/uploads/2024/11/SMCJ-Series.pdf'}
    ]
}
native = M/'eda工程文件/sc-regulator'
sources = [{'path':p.relative_to(M).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
           for p in sorted(native.rglob('*')) if p.is_file()]
result = {'schemaVersion':1,'asOf':'2026-10-21','scope':'现行8S1P原生单页全部166对象及450脚；不含未落图AFE候选',
 'reviewComplete':True,'designRelease':'HOLD','sourceFiles':sources,
 'enet':{'path':enetpath.relative_to(M).as_posix(),'sha256':sha(enetpath),
         'freshExportThisRun':False,'status':'STALE_MATERIAL_METADATA',
         'basis':'保留修改前官方ENET；六件采购/额定属性已更新，旧材料属性不可作为当前采购依据。当前针网/几何与基线逐记录字节相同，见材料变更证明；新官方导出未成功。'},
 'contract':{'components':len(actual),'pins':len(rows),'mismatches':sum(r['status']!='PASS' for r in rows),
             'meaning':'合同一致只证明映射，不证明设计正确'},
 'groups':{k:{'references':v,'pins':sum(len(actual[r]['pinInfoMap']) for r in v)} for k,v in groups.items()},
 'findings':findings,'calculations':calc,'busProtectionReuseScreen':bus_power_screen,
 'api':{'currentSessions':material_proof['sessions'],'readOpenDocument':'REQUEST_TIMEOUT',
        'ownedSessionsClosed':True,'nativeSchematicShaUnchanged':False,
        'changeScope':'Only 48 selected ATTR values; all other native records byte-identical',
        'topologyAndGeometryRecordsByteUnchanged':True,'newDrcStatus':'NOT_VERIFIED',
        'newDrcOrNetlistClaimed':False},
 'materialMetadataUpdate':{'proofPath':material_proof_path.relative_to(M).as_posix(),
        'proofSha256':sha(material_proof_path),'verifiedChangedAttributeRecords':changed_count,
        'selectedPart':material_proof['selectedPart'],'nativeProof':material_proof['nativeProof']},
 'pinMatrix':rows,
 'manufacturerDocuments':[{'manufacturer':'HiLink','mpn':'URB4812YMD-10WR3',
    'localPath':'参考资料/海凌科_URAB_YMD-10WR3_V1.7.pdf',
    'sha256':'1c55e95b4c6457f75984519c61ba2c544b15f9e65016e8ebf4cc286b545ff49d',
    'use':'U_AUX1厂家身份、针脚和电气边界的主依据；本地文件SHA读回确认。'},
   {'manufacturer':'MORNSUN','localPath':'参考资料/MORNSUN_URB_YMD-10WR3_2026.09.17-B5.pdf',
    'use':'不同厂家的同系列资料；不用于U_AUX1身份或规格判定。'}],
 'primarySources':[
 'https://www.ti.com/lit/ds/symlink/drv8161.pdf',
 'https://www.ti.com/lit/ds/symlink/tlv6700.pdf',
 'https://bitwell.cn/Upload/%E4%BF%9D%E6%8A%A4%E8%8A%AF%E7%89%87/BW6101_C%E4%BF%9D%E6%8A%A4IC-09311941072.pdf',
 'https://bitwell.cn/caseview.aspx?CateID=614&ID=122',
 'HiLink URB4812YMD-10WR3原厂资料：参考资料/海凌科_URAB_YMD-10WR3_V1.7.pdf；SHA256 1c55e95b4c6457f75984519c61ba2c544b15f9e65016e8ebf4cc286b545ff49d',
 'https://boardensh.com/wp-content/uploads/2024/11/SMCJ-Series.pdf',
 'https://yageogroup.com/content/datasheet/asset/file/KEM_A4017_PEH220_225_226',
 'https://www.littelfuse.com/~/media/electronics/datasheets/tvs_diodes/littelfuse_tvs_diode_smcj_datasheet.pdf.pdf',
 'https://content.kemet.com/datasheets/KEM_A4034_PEH200.pdf',
 'https://www.st.com/resource/en/datasheet/stm32g431cb.pdf']}
(O/'全功能与逐脚复核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (O/'全部450脚核对.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
md=['# 稳压模块全部原理检查','',
    '2026-10-21。检查覆盖现行单页全部166个对象、450脚，设计合同映射0差异。**全图检查完成，原理放行HOLD**。当前原生仍是8只超容；16只/BQ76952候选没有落图。',
    '', '本轮API切页超时，会话已确认关闭。六只母线bulk已原位改为900µF/80V；独立复核修改前ZIP与当前源，仅48个指定ATTR值变化，其余全部原生记录逐字节相同，故针网/几何未变。旧官方ENET材料属性已过期，新ENET未导出、新DRC未验证。证明见[材料变更证明](原理整改/PEH226NL80V_变更证明.json)。',
    '', '| 功能区 | 对象数 | 引脚数 |','|---|---:|---:|']
for k,v in groups.items():md.append(f'| {k} | {len(v)} | {result["groups"][k]["pins"]} |')
md+=['','## 原理结论与整改边界','']
for f in findings:md += [f'### {f["id"]} · {f["subject"]} · {f["status"]}', '',f['finding'],'',f'处理：{f["action"]}','']
cap = bus_power_screen['bulkCapacitor']
tvss = bus_power_screen['busTvs']['options']
stale_index_modules = [name for name, state in source_freshness.items() if not state['match']]
md += ['## BUS bulk电容与TVS精确复用筛选','',
       f"中央索引SHA256：`{library_index_sha}`。来源SHA未匹配模块：{', '.join(stale_index_modules) if stale_index_modules else '无，四个当前来源均匹配'}。" + ('涉及未匹配来源的用量仅为临时快照。' if stale_index_modules else '当前用量与已同步源绑定；不代替电气资格。'),
       '',
       f"当前稳压原生及BOM已选{cap['current']['referenceCount']}只 KEMET {cap['current']['mpn']}（{cap['current']['capacitanceEachUf']}µF/{cap['current']['ratedVoltageV']}V，合计{cap['current']['totalCapacitanceUf']/1000:.1f}mF）。相对修改前63V型号，数量与容量不变；125°C自然对流、≥5kHz额定纹波由48.6A降到44.4A（每只8.1A降至7.4A），相比40Arms名义余量4.4A，不能据此宣称全温验收。80V器件在150°C热沉条件为5.7A/只、六只34.2A。新件待询价，未沿用旧件价格。",
       '',
       'PEH226 80V器件为同尺寸级L（20.2×43.7mm、15.24mm引脚间距），但本板PCB封装仍标待导入；YAGEO/KEMET 2026-04-28数据表指出L尺寸新设计可能采用top-weld crown，目录件可能为side-weld，必须实物核对焊冠、极性和孔距，不能视作drop-in。PEH226系列最高80V；中央库中的100V PEH200PJ5100MB2为10,000µF、50×95mm螺栓端子，已分配FOC，不是现有径向焊盘的直接替代。',
       '', '| TVS料号 | VRWM | VBR min–max | 最大VC@Ipp | 对37.8V余量 | 索引现有用量 |', '|---|---:|---:|---:|---:|---|']
for option in tvss:
    usage_text = '; '.join(f"{module}×{len(refs)} ({', '.join(refs)})" for module, refs in option['indexUsage']['perModuleDesignators'].items()) or '无'
    if option['indexUsage']['staleSourceModules']:
        usage_text += '；来源待刷新'
    md.append(f"| {option['mpn']} | {option['vrwmV']}V | {option['vbrMinMaxV'][0]}–{option['vbrMinMaxV'][1]}V | {option['vcMaxV']}V@{option['ippA']}A | {option['headroomTo37_8V_V']}V | {usage_text} |")
md += ['', 'SMCJ43A是当前稳压板器件；同型号SMCJ40A可跨模块复用，但正常最高母线到40V VRWM仅2.2V，须确认全容差、纹波和温漂都低于40V。SMCJ45A的VRWM余量更大，钳位也更高。三者的VC均是数据表规定Ipp下的值，不是任意浪涌下的母线绝对上限。Littelfuse SMCJ 1500W指标对应10/1000µs、0.01%占空比非重复脉冲，不能替代实际重复能量/寄生尖峰测量。',
       '', bus_power_screen['busTvs']['screenConclusion'], '']
md += ['## 复算中需要保留的节点边界','',
       '6×900µF位于BUS_EXT，桥侧名义局部电容只有7.7µF；不能把5.4mF当关断后桥侧储能。20µH/80A电感能量64mJ，若全部进入最小160µF的SC侧局部电容，19.6V可理想上升至约34.4V；真实断流路径须单独核对，不能拿35V名义耐压认定有充足裕量。',
       '', 'BW6101的120Ω支路在2.48V下即使忽略内部MOS压降也仅20.7mA/51.3mW。原厂网页200mA与旧PDF700mA为裸芯不同条件；均不能变成本电路支路泄流能力。',
       '', '本轮未修改任何PCB，也未把模型通过改写为实物通过。详细逐脚表和计算见同目录JSON/CSV。', '', '## 原厂资料','']
md += [f'- {url}' for url in result['primarySources']]
(O/'全功能与逐脚复核.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(json.dumps({'objects':len(actual),'pins':len(rows),'mismatches':0,'groups':len(groups),
                  'findings':len(findings),'release':'HOLD','sourceSha':sha(next(native.rglob('*.esch2')))},ensure_ascii=True))
