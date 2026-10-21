#include "速度反馈接口.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

static bool node_is_valid(drive_node_t node)
{
    return node == DRIVE_NODE_LEFT || node == DRIVE_NODE_RIGHT;
}

static uint16_t read_u16_le(const uint8_t *bytes)
{
    return (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
}

static uint32_t read_u32_le(const uint8_t *bytes)
{
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static void write_u16_le(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t)(value & 0xFFu);
    bytes[1] = (uint8_t)(value >> 8);
}

static void write_u32_le(uint8_t *bytes, uint32_t value)
{
    bytes[0] = (uint8_t)(value & 0xFFu);
    bytes[1] = (uint8_t)((value >> 8) & 0xFFu);
    bytes[2] = (uint8_t)((value >> 16) & 0xFFu);
    bytes[3] = (uint8_t)(value >> 24);
}

static int16_t read_i16_le(const uint8_t *bytes)
{
    uint16_t raw = read_u16_le(bytes);
    int32_t value = (raw & 0x8000u) ? ((int32_t)raw - 65536) : (int32_t)raw;
    return (int16_t)value;
}

static drive_result_t frame_prepare(drive_node_t node, uint16_t base_id,
                                    drive_can_frame_t *frame)
{
    if (frame == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    if (!node_is_valid(node)) {
        return DRIVE_ERR_NODE;
    }
    memset(frame, 0, sizeof(*frame));
    frame->id = drive_can_id(node, base_id);
    frame->dlc = DRIVE_CAN_DLC;
    return DRIVE_OK;
}

static drive_result_t frame_validate(const drive_can_frame_t *frame,
                                     drive_node_t node, uint16_t base_id)
{
    uint16_t expected_id;
    uint16_t expected_crc;
    uint16_t received_crc;
    if (frame == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    if (!node_is_valid(node)) {
        return DRIVE_ERR_NODE;
    }
    expected_id = drive_can_id(node, base_id);
    if (frame->id != expected_id || frame->dlc != DRIVE_CAN_DLC ||
        frame->extended || frame->remote) {
        return DRIVE_ERR_FRAME;
    }
    expected_crc = drive_crc16_ccitt_false(frame->id, frame->dlc, frame->data, 6u);
    received_crc = read_u16_le(&frame->data[6]);
    if (expected_crc != received_crc) {
        return DRIVE_ERR_CRC;
    }
    return DRIVE_OK;
}

static void frame_finalize_crc(drive_can_frame_t *frame)
{
    uint16_t crc = drive_crc16_ccitt_false(frame->id, frame->dlc, frame->data, 6u);
    write_u16_le(&frame->data[6], crc);
}

uint16_t drive_crc16_ccitt_false(uint16_t can_id, uint8_t dlc,
                                 const uint8_t *payload, uint8_t payload_length)
{
    uint16_t crc = 0xFFFFu;
    uint8_t input[3u + DRIVE_CAN_DLC];
    uint8_t input_length;
    uint8_t i;
    uint8_t bit;
    if (payload == NULL || payload_length > DRIVE_CAN_DLC) {
        return 0u;
    }
    input[0] = (uint8_t)(can_id & 0xFFu);
    input[1] = (uint8_t)(can_id >> 8);
    input[2] = dlc;
    memcpy(&input[3], payload, payload_length);
    input_length = (uint8_t)(3u + payload_length);
    for (i = 0u; i < input_length; ++i) {
        crc ^= (uint16_t)((uint16_t)input[i] << 8);
        for (bit = 0u; bit < 8u; ++bit) {
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u)
                                  : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

uint16_t drive_can_id(drive_node_t node, uint16_t base_id)
{
    return (uint16_t)(base_id + (uint16_t)node);
}

drive_result_t drive_encode_boot_epoch(drive_node_t node,
                                       const drive_boot_epoch_t *message,
                                       drive_can_frame_t *frame)
{
    drive_result_t result;
    if (message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    if (message->boot_epoch == 0u) {
        return DRIVE_ERR_EPOCH;
    }
    result = frame_prepare(node, DRIVE_CAN_ID_BOOT_BASE, frame);
    if (result != DRIVE_OK) {
        return result;
    }
    write_u32_le(&frame->data[0], message->boot_epoch);
    frame_finalize_crc(frame);
    return DRIVE_OK;
}

drive_result_t drive_decode_boot_epoch(const drive_can_frame_t *frame,
                                       drive_node_t node,
                                       drive_boot_epoch_t *message)
{
    drive_result_t result;
    if (message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    result = frame_validate(frame, node, DRIVE_CAN_ID_BOOT_BASE);
    if (result != DRIVE_OK) {
        return result;
    }
    if (frame->data[4] != 0u || frame->data[5] != 0u) {
        return DRIVE_ERR_RESERVED;
    }
    message->boot_epoch = read_u32_le(&frame->data[0]);
    return message->boot_epoch == 0u ? DRIVE_ERR_EPOCH : DRIVE_OK;
}

static drive_result_t encode_session(drive_node_t node, uint16_t base_id,
                                     const drive_session_t *message,
                                     drive_can_frame_t *frame)
{
    drive_result_t result;
    if (message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    if (message->boot_epoch == 0u || message->session_id == 0u) {
        return DRIVE_ERR_RANGE;
    }
    result = frame_prepare(node, base_id, frame);
    if (result != DRIVE_OK) {
        return result;
    }
    write_u32_le(&frame->data[0], message->boot_epoch);
    write_u16_le(&frame->data[4], message->session_id);
    frame_finalize_crc(frame);
    return DRIVE_OK;
}

static drive_result_t decode_session(const drive_can_frame_t *frame,
                                     drive_node_t node, uint16_t base_id,
                                     drive_session_t *message)
{
    drive_result_t result;
    if (message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    result = frame_validate(frame, node, base_id);
    if (result != DRIVE_OK) {
        return result;
    }
    message->boot_epoch = read_u32_le(&frame->data[0]);
    message->session_id = read_u16_le(&frame->data[4]);
    if (message->boot_epoch == 0u || message->session_id == 0u) {
        return DRIVE_ERR_RANGE;
    }
    return DRIVE_OK;
}

drive_result_t drive_encode_session_begin(drive_node_t node,
                                          const drive_session_t *message,
                                          drive_can_frame_t *frame)
{
    return encode_session(node, DRIVE_CAN_ID_BEGIN_BASE, message, frame);
}

drive_result_t drive_decode_session_begin(const drive_can_frame_t *frame,
                                          drive_node_t node,
                                          drive_session_t *message)
{
    return decode_session(frame, node, DRIVE_CAN_ID_BEGIN_BASE, message);
}

drive_result_t drive_encode_session_close(drive_node_t node,
                                          const drive_session_t *message,
                                          drive_can_frame_t *frame)
{
    return encode_session(node, DRIVE_CAN_ID_CLOSE_BASE, message, frame);
}

drive_result_t drive_decode_session_close(const drive_can_frame_t *frame,
                                          drive_node_t node,
                                          drive_session_t *message)
{
    return decode_session(frame, node, DRIVE_CAN_ID_CLOSE_BASE, message);
}

drive_result_t drive_encode_speed_command(drive_node_t node,
                                          const drive_speed_command_t *message,
                                          drive_can_frame_t *frame)
{
    drive_result_t result;
    if (message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    if (message->session_id == 0u || message->ttl_10ms < DRIVE_SPEED_TTL_MIN_UNITS ||
        message->ttl_10ms > DRIVE_SPEED_TTL_MAX_UNITS ||
        message->target_motor_rpm == INT16_MIN) {
        return DRIVE_ERR_RANGE;
    }
    result = frame_prepare(node, DRIVE_CAN_ID_SPEED_BASE, frame);
    if (result != DRIVE_OK) {
        return result;
    }
    write_u16_le(&frame->data[0], message->session_id);
    frame->data[2] = message->sequence;
    write_u16_le(&frame->data[3], (uint16_t)message->target_motor_rpm);
    frame->data[5] = message->ttl_10ms;
    frame_finalize_crc(frame);
    return DRIVE_OK;
}

drive_result_t drive_decode_speed_command(const drive_can_frame_t *frame,
                                          drive_node_t node,
                                          drive_speed_command_t *message)
{
    drive_result_t result;
    if (message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    result = frame_validate(frame, node, DRIVE_CAN_ID_SPEED_BASE);
    if (result != DRIVE_OK) {
        return result;
    }
    message->session_id = read_u16_le(&frame->data[0]);
    message->sequence = frame->data[2];
    message->target_motor_rpm = read_i16_le(&frame->data[3]);
    message->ttl_10ms = frame->data[5];
    if (message->session_id == 0u || message->ttl_10ms < DRIVE_SPEED_TTL_MIN_UNITS ||
        message->ttl_10ms > DRIVE_SPEED_TTL_MAX_UNITS ||
        message->target_motor_rpm == INT16_MIN) {
        return DRIVE_ERR_RANGE;
    }
    return DRIVE_OK;
}

drive_result_t drive_encode_state(drive_node_t node, const drive_state_t *message,
                                  drive_can_frame_t *frame)
{
    drive_result_t result;
    if (message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    result = frame_prepare(node, DRIVE_CAN_ID_STATE_BASE, frame);
    if (result != DRIVE_OK) {
        return result;
    }
    write_u16_le(&frame->data[0], message->sample_sequence);
    write_u16_le(&frame->data[2], (uint16_t)message->mechanical_rpm);
    write_u16_le(&frame->data[4], (uint16_t)message->iq_centiamp);
    frame_finalize_crc(frame);
    return DRIVE_OK;
}

drive_result_t drive_decode_state(const drive_can_frame_t *frame,
                                  drive_node_t node, drive_state_t *message)
{
    drive_result_t result;
    if (message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    result = frame_validate(frame, node, DRIVE_CAN_ID_STATE_BASE);
    if (result != DRIVE_OK) {
        return result;
    }
    message->sample_sequence = read_u16_le(&frame->data[0]);
    message->mechanical_rpm = read_i16_le(&frame->data[2]);
    message->iq_centiamp = read_i16_le(&frame->data[4]);
    return DRIVE_OK;
}

drive_result_t drive_encode_quality(drive_node_t node,
                                    const drive_quality_t *message,
                                    drive_can_frame_t *frame)
{
    drive_result_t result;
    if (message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    result = frame_prepare(node, DRIVE_CAN_ID_QUALITY_BASE, frame);
    if (result != DRIVE_OK) {
        return result;
    }
    write_u16_le(&frame->data[0], message->sample_sequence);
    write_u16_le(&frame->data[2], message->device_tick_ms_low16);
    frame->data[4] = message->flags;
    frame->data[5] = message->applied_command_sequence;
    frame_finalize_crc(frame);
    return DRIVE_OK;
}

drive_result_t drive_decode_quality(const drive_can_frame_t *frame,
                                    drive_node_t node, drive_quality_t *message)
{
    drive_result_t result;
    if (message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    result = frame_validate(frame, node, DRIVE_CAN_ID_QUALITY_BASE);
    if (result != DRIVE_OK) {
        return result;
    }
    message->sample_sequence = read_u16_le(&frame->data[0]);
    message->device_tick_ms_low16 = read_u16_le(&frame->data[2]);
    message->flags = frame->data[4];
    message->applied_command_sequence = frame->data[5];
    return DRIVE_OK;
}

drive_result_t drive_pair_feedback(const drive_state_t *state,
                                   const drive_quality_t *quality,
                                   drive_feedback_pair_t *pair)
{
    if (state == NULL || quality == NULL || pair == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    if (state->sample_sequence != quality->sample_sequence) {
        return DRIVE_ERR_PAIR;
    }
    if (((quality->flags & DRIVE_QUALITY_RPM_VALID) != 0u &&
         state->mechanical_rpm == INT16_MIN) ||
        ((quality->flags & DRIVE_QUALITY_IQ_VALID) != 0u &&
         state->iq_centiamp == INT16_MIN) ||
        ((quality->flags & DRIVE_QUALITY_RPM_VALID) == 0u &&
         state->mechanical_rpm != INT16_MIN) ||
        ((quality->flags & DRIVE_QUALITY_IQ_VALID) == 0u &&
         state->iq_centiamp != INT16_MIN) ||
        ((quality->flags & DRIVE_QUALITY_APPLIED_COMMAND_VALID) == 0u &&
         quality->applied_command_sequence != 0xFFu)) {
        return DRIVE_ERR_RANGE;
    }
    pair->state = *state;
    pair->quality = *quality;
    return DRIVE_OK;
}

void drive_receiver_init(drive_receiver_t *receiver, drive_node_t node,
                         uint32_t current_boot_epoch)
{
    if (receiver == NULL) {
        return;
    }
    memset(receiver, 0, sizeof(*receiver));
    receiver->node = node;
    receiver->boot_epoch = current_boot_epoch;
}

drive_result_t drive_receiver_begin(drive_receiver_t *receiver,
                                    const drive_session_t *message)
{
    if (receiver == NULL || message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    if (!node_is_valid(receiver->node)) {
        return DRIVE_ERR_NODE;
    }
    if (receiver->boot_epoch == 0u) {
        return DRIVE_ERR_EPOCH;
    }
    if (message->boot_epoch == 0u || message->session_id == 0u) {
        return DRIVE_ERR_RANGE;
    }
    if (message->boot_epoch != receiver->boot_epoch) {
        return DRIVE_ERR_EPOCH;
    }
    if (receiver->session_active) {
        return receiver->active_session_id == message->session_id
                   ? DRIVE_OK
                   : DRIVE_ERR_SESSION;
    }
    if (receiver->have_last_session && message->session_id <= receiver->last_session_id) {
        return DRIVE_ERR_SESSION;
    }
    receiver->session_active = true;
    receiver->active_session_id = message->session_id;
    receiver->last_session_id = message->session_id;
    receiver->have_last_session = true;
    receiver->have_command_sequence = false;
    receiver->command_valid = false;
    memset(&receiver->active_command, 0, sizeof(receiver->active_command));
    return DRIVE_OK;
}

drive_result_t drive_receiver_close(drive_receiver_t *receiver,
                                    const drive_session_t *message)
{
    if (receiver == NULL || message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    if (message->boot_epoch != receiver->boot_epoch) {
        return DRIVE_ERR_EPOCH;
    }
    if (message->session_id == 0u) {
        return DRIVE_ERR_RANGE;
    }
    if (receiver->session_active) {
        if (receiver->active_session_id != message->session_id) {
            return DRIVE_ERR_SESSION;
        }
        receiver->session_active = false;
        receiver->active_session_id = 0u;
        receiver->have_command_sequence = false;
        receiver->command_valid = false;
        memset(&receiver->active_command, 0, sizeof(receiver->active_command));
        return DRIVE_OK;
    }
    return receiver->have_last_session && receiver->last_session_id == message->session_id
               ? DRIVE_OK
               : DRIVE_ERR_SESSION;
}

drive_result_t drive_receiver_accept_speed(drive_receiver_t *receiver,
                                           const drive_speed_command_t *message,
                                           uint32_t receive_time_ms)
{
    uint8_t delta;
    if (receiver == NULL || message == NULL) {
        return DRIVE_ERR_ARGUMENT;
    }
    if (!receiver->session_active || receiver->active_session_id != message->session_id) {
        return DRIVE_ERR_SESSION;
    }
    if (message->ttl_10ms < DRIVE_SPEED_TTL_MIN_UNITS ||
        message->ttl_10ms > DRIVE_SPEED_TTL_MAX_UNITS ||
        message->target_motor_rpm == INT16_MIN) {
        return DRIVE_ERR_RANGE;
    }
    if (receiver->have_command_sequence) {
        delta = (uint8_t)(message->sequence - receiver->last_command_sequence);
        if (delta == 0u || delta >= 128u) {
            return DRIVE_ERR_SEQUENCE;
        }
    }
    receiver->last_command_sequence = message->sequence;
    receiver->have_command_sequence = true;
    receiver->active_command = *message;
    receiver->command_received_ms = receive_time_ms;
    receiver->command_valid = true;
    return DRIVE_OK;
}

bool drive_receiver_poll(drive_receiver_t *receiver, uint32_t now_ms,
                         drive_speed_command_t *command)
{
    uint32_t ttl_ms;
    if (receiver == NULL || !receiver->session_active || !receiver->command_valid) {
        return false;
    }
    ttl_ms = (uint32_t)receiver->active_command.ttl_10ms * DRIVE_SPEED_TTL_UNIT_MS;
    if ((uint32_t)(now_ms - receiver->command_received_ms) >= ttl_ms) {
        receiver->command_valid = false;
        memset(&receiver->active_command, 0, sizeof(receiver->active_command));
        return false;
    }
    if (command != NULL) {
        *command = receiver->active_command;
    }
    return true;
}

bool drive_feedback_is_fresh(uint32_t now_ms, uint32_t host_receive_time_ms,
                             uint32_t max_age_ms)
{
    return (uint32_t)(now_ms - host_receive_time_ms) <= max_age_ms;
}
