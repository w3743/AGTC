/**
 * Copyright (c) NearLink Project. 2026-2026. All rights reserved.
 *
 * Description: SLE Range Link distance filter (median outlier removal + mean). \n
 *              按窗口中值剔除野值，再输出中值（不足三个样本时输出均值）。
 */

#include "range_link_filter.h"
#include "common_def.h"

static int32_t g_range_win[RL_FILTER_WIN];
static uint32_t g_range_win_cnt;
static uint32_t g_range_win_idx;
static uint32_t g_range_valid_total;
static uint32_t g_range_reject_streak;
static int32_t g_range_last_out;

static uint32_t rl_magnitude32(int32_t val)
{
    /* Unsigned negation is defined even for INT32_MIN. */
    return (val < 0) ? (0U - (uint32_t)val) : (uint32_t)val;
}

void rl_filter_init(void)
{
    uint32_t i;

    for (i = 0; i < RL_FILTER_WIN; i++) {
        g_range_win[i] = 0;
    }
    g_range_win_cnt = 0;
    g_range_win_idx = 0;
    g_range_valid_total = 0;
    g_range_reject_streak = 0;
    g_range_last_out = 0;
}

void rl_filter_reset(void)
{
    rl_filter_init();
}

static int32_t rl_filter_median(void)
{
    int32_t tmp[RL_FILTER_WIN];
    uint32_t i;
    uint32_t j;

    if (g_range_win_cnt == 0) {
        return 0;
    }
    for (i = 0; i < g_range_win_cnt; i++) {
        tmp[i] = g_range_win[i];
    }
    /* 插入排序，窗口最大 32，代价可接受 */
    for (i = 1; i < g_range_win_cnt; i++) {
        int32_t key = tmp[i];
        j = i;
        while ((j > 0) && (tmp[j - 1] > key)) {
            tmp[j] = tmp[j - 1];
            j--;
        }
        tmp[j] = key;
    }
    return tmp[g_range_win_cnt / 2];
}

static int32_t rl_filter_window_mean(void)
{
    if (g_range_win_cnt == 0) {
        return 0;
    }
    if (g_range_win_cnt == 1) {
        return g_range_win[0];
    }
    /* Only called when count < 3. Widen before summing two int32 values. */
#if RL_FILTER_WIN > 1
    return (int32_t)(((int64_t)g_range_win[0] + g_range_win[1]) / 2);
#else
    return g_range_win[0];
#endif
}

/* 输出用中值；窗口不足三个样本时输出均值。 */
static int32_t rl_filter_output(void)
{
    return (g_range_win_cnt >= 3) ? rl_filter_median() : rl_filter_window_mean();
}

int32_t rl_filter_push(int32_t raw_mm, uint32_t *valid_num)
{
    int32_t median;
    uint32_t limit;
    uint32_t rate_limit;
    uint32_t diff;
    uint32_t magnitude;

    if (g_range_win_cnt > 0) {
        /* With >=3 entries the last accepted output is this window's median.
         * Rejected samples leave both the window and last output unchanged. */
        median = (g_range_win_cnt >= 3) ? g_range_last_out : rl_filter_median();
        /* Ordered unsigned subtraction represents the full 0..UINT32_MAX gap. */
        diff = (raw_mm >= median) ? (uint32_t)raw_mm - (uint32_t)median :
                                   (uint32_t)median - (uint32_t)raw_mm;
        limit = RL_FILTER_OUTLIER_ABS_MM;
        magnitude = rl_magnitude32(median);
        /* Split before multiplication to avoid overflowing at large magnitudes. */
        rate_limit = (magnitude / 100U) * RL_FILTER_OUTLIER_RATE +
                     ((magnitude % 100U) * RL_FILTER_OUTLIER_RATE) / 100U;
        if (rate_limit > limit) {
            limit = rate_limit;
        }
        if (diff > limit) {
            /* 静态稳定性优先：连续少量野值先"保持上次输出"（不跟着跳），
               只有连续多次超出才认为目标真的移动了，重置窗口跟随。 */
            g_range_reject_streak++;
            if (g_range_reject_streak < RL_FILTER_REJECT_RESET) {
                if (valid_num != NULL) {
                    *valid_num = g_range_valid_total;
                }
                return g_range_last_out;
            }
            g_range_win_cnt = 0;
            g_range_win_idx = 0;
        }
    }
    g_range_reject_streak = 0;

    g_range_win[g_range_win_idx] = raw_mm;
    g_range_win_idx = (g_range_win_idx + 1) % RL_FILTER_WIN;
    if (g_range_win_cnt < RL_FILTER_WIN) {
        g_range_win_cnt++;
    }
    g_range_valid_total++;
    if (valid_num != NULL) {
        *valid_num = g_range_valid_total;
    }
    g_range_last_out = rl_filter_output();
    return g_range_last_out;
}
