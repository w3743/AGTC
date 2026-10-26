/**
 * Copyright (c) NearLink Project. 2026-2026. All rights reserved.
 *
 * Description: Range Link extension layer. \n
 *              All functionality we add on top of the official
 *              application/samples/products/sle_measure_dis sample lives here,
 *              so the official sample files stay as close to upstream as
 *              possible (only a few marked "[RANGE_LINK]" hook lines). \n
 *
 * 依赖关系：
 *   official server_alg.c --(hook)--> rl_ext_on_distance()  距离后处理/校准/打印/notify
 *   official server.c     --(hook)--> rl_ext_on_user_data() 用户数据接收
 *   official client.c     --(hook)--> rl_ext_on_client_msg() 距离 notify 接收
 *   entry sle_range_link.c -> rl_ext_server_init() / rl_ext_client_send_user_data()
 */

#ifndef RL_EXT_H
#define RL_EXT_H

#include <stdint.h>
#include "errcode.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Range Link 自定义 SSAP 消息类型（沿用官方 u32 type; u32 len; u8 data[]） ---- */
#define RL_MSG_DIST         0xFFFFFFE0u   /*!< server -> client: 距离+状态 */
#define RL_MSG_USER_DATA    0xFFFFFFE1u   /*!< client -> server: 用户数据 */

#define RL_USER_PAYLOAD_LEN 12

/* client -> server 用户数据（20 字节） */
typedef struct {
    uint32_t seq;                          /*!< 用户数据包序号，从 0 递增 */
    uint32_t uptime_ms;                    /*!< client 本地上电毫秒 */
    uint8_t  payload[RL_USER_PAYLOAD_LEN]; /*!< 用户数据 */
} rl_user_payload_t;

/* server -> client 距离+状态（36 字节，全整数便于解析） */
typedef struct {
    uint32_t seq;                          /*!< 测距结果序号 */
    int32_t  dist_mm;                      /*!< 校准+滤波后距离 (mm) */
    int32_t  raw_mm;                       /*!< 库原始候选值 (mm) */
    int32_t  prob_pm;                      /*!< 置信度千分比 0..1000 */
    int32_t  rssi_dbm;                     /*!< server 侧 RSSI */
    uint32_t smp_num;                      /*!< 累计有效样本数 */
    uint32_t rx_seq;                       /*!< 最近收到的用户数据 seq */
    uint32_t rx_count;                     /*!< 累计收到用户数据包数 */
    uint32_t rx_len;                       /*!< 最近一包 payload 长度 */
} rl_dist_payload_t;

#if defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER)
/* 先 server.h（里面定义 measure_dis_msg_node_t），再 alg.h（它引用该类型） */
#include "sle_measure_dis_server.h"
#include "sle_measure_dis_server_alg.h"

/* entry 初始化：注册 AT 命令（RDTARE/RDFIT/RDCLR），复位校准状态 */
void rl_ext_server_init(void);

/* IQ 队列积压数（entry 任务里批量排空用） */
uint32_t rl_ext_server_pending_msgs(void);

/* 官方算法算出距离后调用：校准 + 滤波 + 打印 + notify */
void rl_ext_on_distance(measure_dis_profile_msg_dis_t *dis);

/* 收到 client 用户数据时调用 */
void rl_ext_on_user_data(const uint8_t *data, uint32_t len);

/* 当前校准状态（AT 查询 / 调试用） */
int32_t rl_ext_raw_median_mm(void);
int32_t rl_ext_offset_mm(void);
int32_t rl_ext_scale_ppm(void);
#endif /* CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER */

#if defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_CLIENT)
/* 周期向 server 发送用户数据（entry 任务调用） */
int rl_ext_client_send_user_data(void);

/* 收到 server 通知（距离）时调用 */
void rl_ext_on_client_msg(uint8_t *data, uint16_t len);
#endif /* CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_CLIENT */

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif /* RL_EXT_H */
