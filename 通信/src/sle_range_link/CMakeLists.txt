#===============================================================================
# @brief    cmake file of SLE Range Link sample.
#           结构说明：range_link_server/ 与 range_link_client/ 下的文件来自官方
#           samples/products/sle_measure_dis（仅插入少量 [RANGE_LINK] 钩子），
#           我们自己的功能全部在 rl_ext.c / range_link_filter.c 里。
# Copyright (c) NearLink Project. 2026-2026. All rights reserved.
#===============================================================================
if(DEFINED CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_CLIENT)
set(SOURCES_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/range_link_client/sle_measure_dis_client.c
    ${CMAKE_CURRENT_SOURCE_DIR}/range_link_client/sle_measure_dis_client_slem.c
    ${CMAKE_CURRENT_SOURCE_DIR}/rl_ext.c
    ${CMAKE_CURRENT_SOURCE_DIR}/sle_range_link.c
)
set(HEADER_LIST ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/range_link_client)
elseif(DEFINED CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER)
set(SOURCES_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/range_link_server/sle_measure_dis_server.c
    ${CMAKE_CURRENT_SOURCE_DIR}/range_link_server/sle_measure_dis_server_adv.c
    ${CMAKE_CURRENT_SOURCE_DIR}/range_link_server/sle_measure_dis_server_alg.c
    ${CMAKE_CURRENT_SOURCE_DIR}/rl_ext.c
    ${CMAKE_CURRENT_SOURCE_DIR}/range_link_filter.c
    ${CMAKE_CURRENT_SOURCE_DIR}/sle_range_link.c
)
set(HEADER_LIST ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/range_link_server)
endif()

set(SOURCES "${SOURCES}" ${SOURCES_LIST} PARENT_SCOPE)
set(PUBLIC_HEADER "${PUBLIC_HEADER}" ${HEADER_LIST} PARENT_SCOPE)
