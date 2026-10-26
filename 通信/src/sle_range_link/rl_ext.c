/**
 * Copyright (c) NearLink Project. 2026-2026. All rights reserved.
 *
 * Description: Range Link extension layer implementation. \n
 *              官方 sle_measure_dis 示例只做“算距离 + 打印”，本文件补上：
 *                1) 应用层单点/两点线性校准（官方 calib_val 保持 0，避免双重校准）
 *                2) 中值滤波/野值门控/未收敛帧丢弃（range_link_filter.c）
 *                3) 运行中对零与两点拟合（AT+RDTARE / AT+RDFIT / AT+RDCLR）
 *                4) 结构化测距打印与可选 notify
 *                5) 双向用户数据传输（client -> server）/ 距离回传（server -> client）
 *
 * History: \n
 * 2026-10-25, Create file. \n
 */
#include <stddef.h>
#include "rl_ext.h"
#include "soc_osal.h"
#include "common_def.h"
#include "tcxo.h"
#include "range_link_filter.h"
#if defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER)
#include "sle_measure_dis_server.h"
#include "at.h"
#elif defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_CLIENT)
#include "sle_measure_dis_client.h"
#endif

#define RL_DIST_REPORT_MS   500            /* 打印节流 */
#define RL_DIST_NOTIFY_MS   1000           /* notify 节流（与 IQ 上传共用连接，降频） */
#define RL_RAW_HIST         16             /* 原始距离历史长度（校准/拟合用） */

/* ==========================================================================
 * server 侧
 * ========================================================================== */
#if defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER)

extern unsigned long g_measure_dis_queue;

static int32_t g_rl_raw_hist[RL_RAW_HIST];
static uint32_t g_rl_raw_cnt = 0;
static uint32_t g_rl_raw_idx = 0;
static int32_t g_rl_offset_mm = CONFIG_RANGE_LINK_CALIB_MM;
static int32_t g_rl_scale_ppm = CONFIG_RANGE_LINK_CALIB_SCALE_PM;
static uint8_t g_rl_auto_tare_done = 0;
static uint8_t g_rl_manual_tare = 0;
static volatile int32_t g_rl_tare_req = 0;            /* 0=无请求；否则 target+1 */
static volatile int32_t g_rl_fit_scale_req = 0;
static volatile int32_t g_rl_fit_offset_req = 0;
static volatile uint8_t g_rl_fit_req = 0;

static uint32_t g_rl_dist_seq = 0;
static uint32_t g_rl_rx_seq = 0;
static uint32_t g_rl_rx_count = 0;
static uint32_t g_rl_rx_len = 0;
static uint32_t g_rl_last_report_ms = 0;
static uint32_t g_rl_last_notify_ms = 0;

/* ---- 两点/多点线性拟合：raw = a*true + b，输出用 cal = raw/a - b/a ---- */
#define RL_FIT_MAX_POINTS 4
typedef struct {
    int32_t true_mm;
    int32_t raw_mm;
} rl_fit_pt_t;
static rl_fit_pt_t g_rl_fit_pts[RL_FIT_MAX_POINTS];
static uint32_t g_rl_fit_cnt = 0;

static int32_t rl_median_i32(const int32_t *buf, uint32_t n)
{
    int32_t tmp[RL_RAW_HIST];
    uint32_t i;
    uint32_t j;

    if ((buf == NULL) || (n == 0) || (n > RL_RAW_HIST)) {
        return 0;
    }
    for (i = 0; i < n; i++) {
        tmp[i] = buf[i];
    }
    for (i = 1; i < n; i++) {
        int32_t key = tmp[i];
        j = i;
        while ((j > 0) && (tmp[j - 1] > key)) {
            tmp[j] = tmp[j - 1];
            j--;
        }
        tmp[j] = key;
    }
    return tmp[n / 2];
}

static void rl_raw_hist_push(int32_t d1_mm)
{
    g_rl_raw_hist[g_rl_raw_idx] = d1_mm;
    g_rl_raw_idx = (g_rl_raw_idx + 1) % RL_RAW_HIST;
    if (g_rl_raw_cnt < RL_RAW_HIST) {
        g_rl_raw_cnt++;
    }
}

int32_t rl_ext_raw_median_mm(void)
{
    return (g_rl_raw_cnt == 0) ? 0 : rl_median_i32(g_rl_raw_hist, g_rl_raw_cnt);
}

int32_t rl_ext_offset_mm(void)
{
    return g_rl_offset_mm;
}

int32_t rl_ext_scale_ppm(void)
{
    return g_rl_scale_ppm;
}

/* 对零：把当前原始中位数当成 target_mm，反推偏移（保留当前刻度） */
static void rl_tare(int32_t target_mm)
{
    int32_t med;

    if (g_rl_raw_cnt == 0) {
        osal_printk("[RL] tare: no raw samples yet, keep offset=%dmm\r\n", (int)g_rl_offset_mm);
        return;
    }
    med = rl_median_i32(g_rl_raw_hist, g_rl_raw_cnt);
    if (target_mm < 0) {
        target_mm = 0;
    }
    g_rl_offset_mm = (int32_t)(((int64_t)med * g_rl_scale_ppm) / 1000) - target_mm;
    g_rl_manual_tare = 1;
    g_rl_auto_tare_done = 1;
    rl_filter_reset();
    osal_printk("[RL] tare ok: raw_med=%dmm target=%dmm scale=%dpermille offset=%dmm\r\n",
                (int)med, (int)target_mm, (int)g_rl_scale_ppm, (int)g_rl_offset_mm);
}

/* 最小二乘拟合并应用（scale_ppm = 1000/a，offset = b/a） */
static void rl_fit_apply(void)
{
    int64_t sx = 0;
    int64_t sy = 0;
    int64_t sxx = 0;
    int64_t sxy = 0;
    int64_t num;
    int64_t den;
    int64_t a_milli;
    int64_t b_mm;
    int64_t scale_ppm;
    int64_t offset_mm;
    int32_t n = (int32_t)g_rl_fit_cnt;
    int32_t i;

    if (n < 2) {
        osal_printk("[RL] fit: need >=2 points, have %d\r\n", (int)n);
        return;
    }
    for (i = 0; i < n; i++) {
        int64_t x = (int64_t)g_rl_fit_pts[i].true_mm;
        int64_t y = (int64_t)g_rl_fit_pts[i].raw_mm;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    den = (int64_t)n * sxx - sx * sx;
    num = (int64_t)n * sxy - sx * sy;
    if ((den <= 0) || (num <= 0)) {
        osal_printk("[RL] fit: two different distances required\r\n");
        return;
    }
    a_milli = (num * 1000) / den;
    if ((a_milli < 300) || (a_milli > 3000)) {
        osal_printk("[RL] fit: slope out of range (raw=%d.%03d*true), check points\r\n",
                    (int)(a_milli / 1000), (int)(a_milli % 1000));
        return;
    }
    scale_ppm = (1000 * den) / num;
    offset_mm = (sy * den - num * sx) / ((int64_t)n * num);
    b_mm = (sy * den - num * sx) / ((int64_t)n * den);
    /* 只登记结果，由测距任务在同一上下文应用（避免跨任务改滤波窗口/打印上下文） */
    g_rl_fit_scale_req = (int32_t)scale_ppm;
    g_rl_fit_offset_req = (int32_t)offset_mm;
    g_rl_fit_req = 1;
    osal_printk("[RL] fit ok: SCALE=%dpermille OFFSET=%dmm (raw=%d.%03d*true+%dmm, points=%d)\r\n",
                (int)scale_ppm, (int)offset_mm, (int)(a_milli / 1000), (int)(a_milli % 1000),
                (int)b_mm, (int)n);
}

static void rl_add_fit_point(int32_t true_mm)
{
    int32_t raw = rl_ext_raw_median_mm();
    int32_t i;

    if (raw <= 0) {
        osal_printk("[RL] fit: no raw samples yet, let it run ~10s first\r\n");
        return;
    }
    if (g_rl_fit_cnt < RL_FIT_MAX_POINTS) {
        g_rl_fit_pts[g_rl_fit_cnt].true_mm = true_mm;
        g_rl_fit_pts[g_rl_fit_cnt].raw_mm = raw;
        g_rl_fit_cnt++;
    } else {
        for (i = 1; i < RL_FIT_MAX_POINTS; i++) {
            g_rl_fit_pts[i - 1] = g_rl_fit_pts[i];
        }
        g_rl_fit_pts[RL_FIT_MAX_POINTS - 1].true_mm = true_mm;
        g_rl_fit_pts[RL_FIT_MAX_POINTS - 1].raw_mm = raw;
    }
    osal_printk("[RL] fit point %d: true=%dmm raw_med=%dmm\r\n", (int)g_rl_fit_cnt, (int)true_mm, (int)raw);
    if (g_rl_fit_cnt >= 2) {
        rl_fit_apply();
    }
}

/* 应用 AT 任务登记的请求（在测距任务上下文执行，避免跨任务改滤波窗口） */
static void rl_apply_pending(void)
{
    if (g_rl_tare_req != 0) {
        int32_t target = g_rl_tare_req - 1;
        g_rl_tare_req = 0;
        rl_tare(target);
    }
    if (g_rl_fit_req != 0) {
        g_rl_fit_req = 0;
        g_rl_scale_ppm = g_rl_fit_scale_req;
        g_rl_offset_mm = g_rl_fit_offset_req;
        g_rl_manual_tare = 1;
        g_rl_auto_tare_done = 1;
        rl_filter_reset();
        osal_printk("[RL] fit applied: scale=%dpermille offset=%dmm\r\n",
                    (int)g_rl_scale_ppm, (int)g_rl_offset_mm);
    }
}

uint32_t rl_ext_server_pending_msgs(void)
{
    return (uint32_t)osal_msg_queue_get_msg_num(g_measure_dis_queue);
}

void rl_ext_on_distance(measure_dis_profile_msg_dis_t *dis)
{
    uint32_t now;
    uint32_t valid_num = 0;
    int32_t d1_mm;
    int32_t d2_mm;
    int32_t cal_mm;
    int32_t prob_pm;
    uint8_t do_print;
    rl_dist_payload_t pkt;

    if (dis == NULL) {
        return;
    }
    d1_mm = (int32_t)(dis->dist_first * MEASURE_DIS_NUM_CARRY_1000);
    d2_mm = (int32_t)(dis->dist_second * MEASURE_DIS_NUM_CARRY_1000);
    if (d1_mm <= 0) {
        return;     /* 库返回错误（模板值 -1）→ 丢弃 */
    }
    prob_pm = (int32_t)(dis->prob * MEASURE_DIS_NUM_CARRY_1000);
    if (prob_pm < CONFIG_RANGE_LINK_PROB_MIN) {
        return;     /* 置信度过低 */
    }
    if (dis->smooth_num < 3) {
        return;     /* 库内 Kalman 尚未收敛 */
    }
    if ((int32_t)(dis->rssi) < -70) {
        return;     /* 链路太弱，相位测距不可靠 */
    }

    if (dis->smooth_num >= 6) {
        rl_raw_hist_push(d1_mm);
    }
    rl_apply_pending();

#if (CONFIG_RANGE_LINK_AUTO_TARE_MM > 0)
    /* 开机自动对零：收满 16 帧收敛样本后按“两板当前间距 = AUTO_TARE_MM”对零 */
    if ((g_rl_auto_tare_done == 0) && (g_rl_manual_tare == 0) && (g_rl_raw_cnt >= RL_RAW_HIST)) {
        int32_t med = rl_median_i32(g_rl_raw_hist, RL_RAW_HIST);
        g_rl_offset_mm = (int32_t)(((int64_t)med * g_rl_scale_ppm) / 1000) - CONFIG_RANGE_LINK_AUTO_TARE_MM;
        g_rl_auto_tare_done = 1;
        osal_printk("[RL] auto-tare D1_med=%dmm target=%dmm offset=%dmm\r\n",
                    (int)med, (int)CONFIG_RANGE_LINK_AUTO_TARE_MM, (int)g_rl_offset_mm);
    }
#endif

    cal_mm = (int32_t)(((int64_t)d1_mm * g_rl_scale_ppm) / 1000) - g_rl_offset_mm;
    if (cal_mm < 0) {
        cal_mm = 0;   /* 近距离时原始值可能小于偏移 → 钳到 0，绝不输出负距离 */
    }
    valid_num = 0;
    cal_mm = rl_filter_push(cal_mm, &valid_num);
    if (cal_mm < 0) {
        cal_mm = 0;
    }

    now = (uint32_t)uapi_tcxo_get_ms();
    do_print = ((g_rl_last_report_ms == 0) || ((now - g_rl_last_report_ms) >= RL_DIST_REPORT_MS));
    if (do_print) {
        g_rl_last_report_ms = now;
    }

    pkt.seq = ++g_rl_dist_seq;
    pkt.dist_mm = cal_mm;
    pkt.raw_mm = d2_mm;
    pkt.prob_pm = prob_pm;
    pkt.rssi_dbm = (int32_t)(dis->rssi);
    pkt.smp_num = valid_num;
    pkt.rx_seq = g_rl_rx_seq;
    pkt.rx_count = g_rl_rx_count;
    pkt.rx_len = g_rl_rx_len;

    if (do_print) {
        /* D=校准+滤波后距离 D1/D2=库原始值（未校准） SN=库平滑帧数 SC/OF=当前校准参数 */
        osal_printk("[RL] server D=%dmm D1=%dmm D2=%dmm PROB=%d SN=%u RSSI=%ddBm SC=%d OF=%d RX_SEQ=%u RX_CNT=%u\r\n",
                    (int)cal_mm, (int)d1_mm, (int)d2_mm, (int)prob_pm,
                    (unsigned)dis->smooth_num, (int)pkt.rssi_dbm,
                    (int)g_rl_scale_ppm, (int)g_rl_offset_mm,
                    (unsigned)g_rl_rx_seq, (unsigned)g_rl_rx_count);
    }

#ifdef CONFIG_RANGE_LINK_DIST_NOTIFY
    /* notify 与测距共用同一条连接：降频到 1s 一次，减少对 IQ 上传的干扰 */
    if ((g_rl_last_notify_ms == 0) || ((now - g_rl_last_notify_ms) >= RL_DIST_NOTIFY_MS)) {
        g_rl_last_notify_ms = now;
        (void)measure_dis_server_write_client(RL_MSG_DIST, (uint8_t *)&pkt, (uint32_t)sizeof(pkt));
    }
#endif
}

void rl_ext_on_user_data(const uint8_t *data, uint32_t len)
{
    const rl_user_payload_t *pkt = (const rl_user_payload_t *)data;

    if ((data == NULL) || (len < sizeof(rl_user_payload_t))) {
        return;
    }
    g_rl_rx_seq = pkt->seq;
    g_rl_rx_count++;
    g_rl_rx_len = len;
#ifdef CONFIG_RANGE_LINK_RX_PRINT
    osal_printk("[RL] server RX seq=%u len=%u\r\n", (unsigned)pkt->seq, (unsigned)len);
#endif
}

/* ---------------- AT 命令：运行时校准（不用重烧固件） ---------------- */
typedef struct {
    uint32_t para_map;
    int32_t mm;
} rl_at_args_t;

static const at_para_parse_syntax_t g_rl_at_para[] = {
    {
        .type = AT_SYNTAX_TYPE_INT,
        .last = true,
        .attribute = AT_SYNTAX_ATTR_AT_MIN_VALUE | AT_SYNTAX_ATTR_AT_MAX_VALUE,
        .entry.int_range.min_val = 0,
        .entry.int_range.max_val = 10000,
        .offset = offsetof(rl_at_args_t, mm),
    },
};

static at_ret_t rl_at_set_tare(const rl_at_args_t *arg)
{
    if (arg == NULL) {
        return AT_RET_CMD_PARA_ERROR;
    }
    g_rl_tare_req = arg->mm + 1;
    osal_printk("[RL] AT+RDTARE=%dmm accepted\r\n", (int)arg->mm);
    return AT_RET_OK;
}

static at_ret_t rl_at_read_tare(void)
{
    osal_printk("+RDTARE:%d,%d,%d\r\n", (int)rl_ext_raw_median_mm(),
                (int)g_rl_offset_mm, (int)g_rl_scale_ppm);
    return AT_RET_OK;
}

static at_ret_t rl_at_set_fit(const rl_at_args_t *arg)
{
    if (arg == NULL) {
        return AT_RET_CMD_PARA_ERROR;
    }
    rl_add_fit_point(arg->mm);   /* 命令上下文只记录/计算，应用交给测距任务 */
    return AT_RET_OK;
}

static at_ret_t rl_at_read_fit(void)
{
    int32_t i;

    osal_printk("+RDFIT:%d,%d,%d", (int)g_rl_fit_cnt, (int)g_rl_scale_ppm, (int)g_rl_offset_mm);
    for (i = 0; i < (int32_t)g_rl_fit_cnt; i++) {
        osal_printk(",%d,%d", (int)g_rl_fit_pts[i].true_mm, (int)g_rl_fit_pts[i].raw_mm);
    }
    osal_printk("\r\n");
    return AT_RET_OK;
}

static at_ret_t rl_at_clear_fit(const rl_at_args_t *arg)
{
    if (arg == NULL) {
        return AT_RET_CMD_PARA_ERROR;
    }
    g_rl_fit_cnt = 0;
    osal_printk("[RL] fit points cleared\r\n");
    return AT_RET_OK;
}

static const at_cmd_entry_t g_rl_at_table[] = {
    {
        "RDTARE", 0x0201, 0, g_rl_at_para, NULL,
        (at_set_func_t)rl_at_set_tare, (at_read_func_t)rl_at_read_tare, NULL,
    },
    {
        "RDFIT", 0x0202, 0, g_rl_at_para, NULL,
        (at_set_func_t)rl_at_set_fit, (at_read_func_t)rl_at_read_fit, NULL,
    },
    {
        "RDCLR", 0x0203, 0, g_rl_at_para, NULL,
        (at_set_func_t)rl_at_clear_fit, NULL, NULL,
    },
};

void rl_ext_server_init(void)
{
    uint32_t ret;

    rl_filter_init();
    ret = (uint32_t)uapi_at_cmd_table_register(g_rl_at_table,
                                               (uint32_t)(sizeof(g_rl_at_table) / sizeof(g_rl_at_table[0])),
                                               (uint32_t)sizeof(rl_at_args_t));
    osal_printk("[RL] AT cmd register ret=0x%x: RDTARE/RDFIT/RDCLR\r\n", (unsigned)ret);
}

#endif /* CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER */

/* ==========================================================================
 * client 侧
 * ========================================================================== */
#if defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_CLIENT)

static uint32_t g_rl_tx_seq = 0;

int rl_ext_client_send_user_data(void)
{
    rl_user_payload_t pkt;
    uint32_t i;
    int ret;

    pkt.seq = ++g_rl_tx_seq;
    pkt.uptime_ms = (uint32_t)uapi_tcxo_get_ms();
    for (i = 0; i < RL_USER_PAYLOAD_LEN; i++) {
        pkt.payload[i] = (uint8_t)(pkt.seq + i);
    }
    ret = measure_dis_client_write_server(RL_MSG_USER_DATA, (uint8_t *)&pkt, (uint32_t)sizeof(pkt));
#ifdef CONFIG_RANGE_LINK_RX_PRINT
    osal_printk("[RL] client TX seq=%u len=%u ret=%d\r\n", (unsigned)pkt.seq, (unsigned)sizeof(pkt), ret);
#else
    (void)ret;
#endif
    return ret;
}

void rl_ext_on_client_msg(uint8_t *data, uint16_t len)
{
    const rl_dist_payload_t *pkt = (const rl_dist_payload_t *)data;

    if ((data == NULL) || (len < sizeof(rl_dist_payload_t))) {
        return;
    }
    osal_printk("[RL] client D=%dmm PROB=%d RSSI=%d N=%u\r\n",
                (int)pkt->dist_mm, (int)pkt->prob_pm, (int)pkt->rssi_dbm, (unsigned)pkt->smp_num);
}

#endif /* CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_CLIENT */
