/**
 * Copyright (c) NearLink Project. 2026-2026. All rights reserved.
 *
 * Description: SLE Range Link distance filter (median outlier removal + mean). \n
 */

#ifndef RANGE_LINK_FILTER_H
#define RANGE_LINK_FILTER_H

#include <stdint.h>

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif /* __cplusplus */
#endif /* __cplusplus */

/* 滤波窗口默认值，可由 Kconfig(CONFIG_RANGE_LINK_FILTER_WIN) 覆盖 */
#ifdef CONFIG_RANGE_LINK_FILTER_WIN
#define RL_FILTER_WIN CONFIG_RANGE_LINK_FILTER_WIN
#else
#define RL_FILTER_WIN 8
#endif

#define RL_FILTER_OUTLIER_ABS_MM 500        /* 野值绝对门限 (mm) */
#define RL_FILTER_OUTLIER_RATE   30         /* 野值相对门限 (%) */
#define RL_FILTER_REJECT_RESET   12         /* 连续 N 次判野值才重置窗口（静态稳定性优先） */

void rl_filter_init(void);

/**
 * @brief 清空滤波窗口（运行中重新对零后调用，避免旧窗口值残留）。
 */
void rl_filter_reset(void);

/**
 * @brief 压入一个原始距离样本，返回窗口均值。
 * @param raw_mm      原始距离 (mm)
 * @param valid_num   输出：累计有效样本数
 * @retval            滤波后距离 (mm)
 */
int32_t rl_filter_push(int32_t raw_mm, uint32_t *valid_num);

#ifdef __cplusplus
#if __cplusplus
}
#endif /* __cplusplus */
#endif /* __cplusplus */

#endif /* RANGE_LINK_FILTER_H */
