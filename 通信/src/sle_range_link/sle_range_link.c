/**
 * Copyright (c) NearLink Project. 2026-2026. All rights reserved.
 *
 * Description: SLE Range Link sample entry. \n
 *              基于官方 application/samples/products/sle_measure_dis/sle_measure_dis.c，
 *              只做三件事（其余功能在 rl_ext.c，官方 server/client/alg 文件保持原样）：
 *                1) server 任务循环里每轮喂狗（SDK 只在 idle 任务喂狗，连续测距会饿死 idle）
 *                2) 初始化扩展层（注册 AT 校准命令 RDTARE/RDFIT/RDCLR）
 *                3) client 任务周期发送用户数据
 *
 * History: \n
 * 2026-10-25, Create file (refactored to reuse official sample). \n
 */
#include "soc_osal.h"
#include "common_def.h"
#include "app_init.h"
#include "watchdog.h"
#include "rl_ext.h"
#if defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER)
#include "sle_measure_dis_server.h"
#elif defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_CLIENT)
#include "sle_measure_dis_client.h"
#endif

#define TASK_PRIORITY_HADM          27
#define STACK_SIZE_BASELINE 0x200
#if defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER)
#define BTH_HADM_SERVICE_STACK_SIZE (STACK_SIZE_BASELINE * 4 + 0x1000)
#else
#define BTH_HADM_SERVICE_STACK_SIZE (STACK_SIZE_BASELINE * 4 + 0x200)
#endif

#if defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER)
void sle_range_link_server_task(void)
{
    uint32_t ret;
    uint32_t i;

    ret = measure_dis_server_init();
    check_rc_return_rc(ret, "con sle_measure_dis_server_init");
    rl_ext_server_init();

    while (1) {
        ret = osal_event_read(
            &measure_dis_evt, MEASURE_DIS_MSG_EVENT,
            OSAL_WAIT_FOREVER, OSAL_WAITMODE_OR | OSAL_WAITMODE_CLR);

        /* 队列可能一次积压多条 IQ 消息：每批最多排空 8 条，避免队列溢出导致丢测距/打印风暴 */
        for (i = 0; i < 8; i++) {
            if (rl_ext_server_pending_msgs() == 0) {
                break;
            }
            sle_measure_dis_msg_proc();
        }

        /* 看门狗只在 idle 任务里喂狗：本任务连续处理 IQ 时必须每轮主动喂狗，
           否则 idle 被饿死 → 8s 后 CWDT 复位（实测连续测距时队列长期非空）。
           仅在队列空闲时让出 1ms，连续测量时保持官方处理节奏。 */
        (void)uapi_watchdog_kick();
        if (rl_ext_server_pending_msgs() == 0) {
            osal_msleep(1);
        }
    }
}
#elif defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_CLIENT)
void sle_range_link_client_task(void)
{
    uint32_t ret;

    ret = measure_dis_client_init();
    check_rc_return_rc(ret, "con sle_measure_dis_client_init");
    while (1) {
        /* 周期向 server 发送用户数据（RL_MSG_USER_DATA），默认 5s 一次 */
        osal_msleep(CONFIG_RANGE_LINK_USER_PERIOD_MS);
        (void)rl_ext_client_send_user_data();
    }
}
#endif

static void sle_range_link_entry(void)
{
#if defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER)
    osal_printk("[RL] boot role=SERVER build=%s %s\r\n", __DATE__, __TIME__);
#else
    osal_printk("[RL] boot role=CLIENT build=%s %s\r\n", __DATE__, __TIME__);
#endif
    osal_task *task_handle = NULL;
    osal_kthread_lock();
#if defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER)
    task_handle = osal_kthread_create((osal_kthread_handler)sle_range_link_server_task,
                                      0, "SLE_RangeLinkSrv", BTH_HADM_SERVICE_STACK_SIZE);
#elif defined(CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_CLIENT)
    task_handle = osal_kthread_create((osal_kthread_handler)sle_range_link_client_task,
                                      0, "SLE_RangeLinkCli", BTH_HADM_SERVICE_STACK_SIZE);
#endif
    if (task_handle != NULL) {
        osal_kthread_set_priority(task_handle, TASK_PRIORITY_HADM);
        osal_kfree(task_handle);
    }
    osal_kthread_unlock();
}

/* Run the sle_range_link_entry. */
app_run(sle_range_link_entry);
