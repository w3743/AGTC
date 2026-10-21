#ifndef DRIVE_SPEED_FEEDBACK_INTERFACE_H
#define DRIVE_SPEED_FEEDBACK_INTERFACE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DRIVE_CAN_DLC 8u
#define DRIVE_PROTOCOL_VERSION 1u
#define DRIVE_SPEED_TTL_UNIT_MS 10u
#define DRIVE_SPEED_TTL_MIN_UNITS 1u
#define DRIVE_SPEED_TTL_MAX_UNITS 10u
#define DRIVE_FEEDBACK_STALE_LIMIT_MS 100u
#define DRIVE_IQ_CENTIAMP_PER_AMP 100

#define DRIVE_CAN_ID_BOOT_BASE 0x04D0u
#define DRIVE_CAN_ID_BEGIN_BASE 0x04E0u
#define DRIVE_CAN_ID_CLOSE_BASE 0x04F0u
#define DRIVE_CAN_ID_SPEED_BASE 0x0500u
#define DRIVE_CAN_ID_STATE_BASE 0x0520u
#define DRIVE_CAN_ID_QUALITY_BASE 0x0540u

typedef enum {
    DRIVE_NODE_LEFT = 1,
    DRIVE_NODE_RIGHT = 2
} drive_node_t;

typedef enum {
    DRIVE_OK = 0,
    DRIVE_ERR_ARGUMENT,
    DRIVE_ERR_NODE,
    DRIVE_ERR_FRAME,
    DRIVE_ERR_CRC,
    DRIVE_ERR_VERSION,
    DRIVE_ERR_RESERVED,
    DRIVE_ERR_RANGE,
    DRIVE_ERR_EPOCH,
    DRIVE_ERR_SESSION,
    DRIVE_ERR_SEQUENCE,
    DRIVE_ERR_PAIR
} drive_result_t;

typedef struct {
    uint16_t id;
    uint8_t dlc;
    uint8_t data[DRIVE_CAN_DLC];
    bool extended;
    bool remote;
} drive_can_frame_t;

typedef struct {
    uint32_t boot_epoch;
} drive_boot_epoch_t;

typedef struct {
    uint32_t boot_epoch;
    uint16_t session_id;
} drive_session_t;

typedef struct {
    uint16_t session_id;
    uint8_t sequence;
    int16_t target_motor_rpm;
    uint8_t ttl_10ms;
} drive_speed_command_t;

typedef struct {
    uint16_t sample_sequence;
    int16_t mechanical_rpm;
    int16_t iq_centiamp;
} drive_state_t;

enum {
    DRIVE_QUALITY_HALL_VALID = 1u << 0,
    DRIVE_QUALITY_RPM_VALID = 1u << 1,
    DRIVE_QUALITY_IQ_VALID = 1u << 2,
    DRIVE_QUALITY_ELECTRICAL_ANGLE_VALID = 1u << 3,
    DRIVE_QUALITY_LOW_SPEED_OR_REVERSAL_DEGRADED = 1u << 4,
    DRIVE_QUALITY_LOCAL_LIMIT_ACTIVE = 1u << 5,
    DRIVE_QUALITY_FAULT_LATCHED = 1u << 6,
    DRIVE_QUALITY_APPLIED_COMMAND_VALID = 1u << 7
};

typedef struct {
    uint16_t sample_sequence;
    uint16_t device_tick_ms_low16;
    uint8_t flags;
    uint8_t applied_command_sequence;
} drive_quality_t;

typedef struct {
    drive_state_t state;
    drive_quality_t quality;
} drive_feedback_pair_t;

typedef struct {
    drive_node_t node;
    uint32_t boot_epoch;
    uint16_t active_session_id;
    uint16_t last_session_id;
    uint8_t last_command_sequence;
    uint32_t command_received_ms;
    drive_speed_command_t active_command;
    bool session_active;
    bool have_last_session;
    bool have_command_sequence;
    bool command_valid;
} drive_receiver_t;

uint16_t drive_crc16_ccitt_false(uint16_t can_id, uint8_t dlc,
                                 const uint8_t *payload, uint8_t payload_length);
uint16_t drive_can_id(drive_node_t node, uint16_t base_id);

drive_result_t drive_encode_boot_epoch(drive_node_t node,
                                       const drive_boot_epoch_t *message,
                                       drive_can_frame_t *frame);
drive_result_t drive_decode_boot_epoch(const drive_can_frame_t *frame,
                                       drive_node_t node,
                                       drive_boot_epoch_t *message);

drive_result_t drive_encode_session_begin(drive_node_t node,
                                          const drive_session_t *message,
                                          drive_can_frame_t *frame);
drive_result_t drive_decode_session_begin(const drive_can_frame_t *frame,
                                          drive_node_t node,
                                          drive_session_t *message);
drive_result_t drive_encode_session_close(drive_node_t node,
                                          const drive_session_t *message,
                                          drive_can_frame_t *frame);
drive_result_t drive_decode_session_close(const drive_can_frame_t *frame,
                                          drive_node_t node,
                                          drive_session_t *message);

drive_result_t drive_encode_speed_command(drive_node_t node,
                                          const drive_speed_command_t *message,
                                          drive_can_frame_t *frame);
drive_result_t drive_decode_speed_command(const drive_can_frame_t *frame,
                                          drive_node_t node,
                                          drive_speed_command_t *message);

drive_result_t drive_encode_state(drive_node_t node, const drive_state_t *message,
                                  drive_can_frame_t *frame);
drive_result_t drive_decode_state(const drive_can_frame_t *frame,
                                  drive_node_t node, drive_state_t *message);
drive_result_t drive_encode_quality(drive_node_t node,
                                    const drive_quality_t *message,
                                    drive_can_frame_t *frame);
drive_result_t drive_decode_quality(const drive_can_frame_t *frame,
                                    drive_node_t node, drive_quality_t *message);
drive_result_t drive_pair_feedback(const drive_state_t *state,
                                   const drive_quality_t *quality,
                                   drive_feedback_pair_t *pair);

void drive_receiver_init(drive_receiver_t *receiver, drive_node_t node,
                         uint32_t current_boot_epoch);
drive_result_t drive_receiver_begin(drive_receiver_t *receiver,
                                    const drive_session_t *message);
drive_result_t drive_receiver_close(drive_receiver_t *receiver,
                                    const drive_session_t *message);
drive_result_t drive_receiver_accept_speed(drive_receiver_t *receiver,
                                           const drive_speed_command_t *message,
                                           uint32_t receive_time_ms);
bool drive_receiver_poll(drive_receiver_t *receiver, uint32_t now_ms,
                         drive_speed_command_t *command);
bool drive_feedback_is_fresh(uint32_t now_ms, uint32_t host_receive_time_ms,
                             uint32_t max_age_ms);

#ifdef __cplusplus
}
#endif

#endif
