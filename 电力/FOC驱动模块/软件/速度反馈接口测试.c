#include "速度反馈接口.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

static unsigned int checks;
static unsigned int failures;

#define CHECK(expr) do { \
    ++checks; \
    if (!(expr)) { \
        ++failures; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

static bool frame_is(const drive_can_frame_t *frame, uint16_t id,
                     const uint8_t expected[DRIVE_CAN_DLC])
{
    return frame->id == id && frame->dlc == DRIVE_CAN_DLC &&
           !frame->extended && !frame->remote &&
           memcmp(frame->data, expected, DRIVE_CAN_DLC) == 0;
}

static void frame_refresh_crc(drive_can_frame_t *frame)
{
    uint16_t crc = drive_crc16_ccitt_false(frame->id, frame->dlc, frame->data, 6u);
    frame->data[6] = (uint8_t)(crc & 0xFFu);
    frame->data[7] = (uint8_t)(crc >> 8);
}

static void test_golden_vectors(void)
{
    static const uint8_t boot_data[8] = {0xEF, 0xCD, 0xAB, 0x89, 0x00, 0x00, 0x5F, 0xD1};
    static const uint8_t begin_data[8] = {0xEF, 0xCD, 0xAB, 0x89, 0x34, 0x12, 0x7F, 0xDE};
    static const uint8_t close_data[8] = {0xEF, 0xCD, 0xAB, 0x89, 0x34, 0x12, 0x81, 0x8D};
    static const uint8_t speed_left_data[8] = {0x34, 0x12, 0x7E, 0xB0, 0x04, 0x0A, 0xE8, 0x32};
    static const uint8_t speed_right_data[8] = {0xEF, 0xBE, 0x00, 0x24, 0xFA, 0x05, 0x02, 0x5C};
    static const uint8_t state_data[8] = {0x44, 0x22, 0x2E, 0xFB, 0x57, 0xFE, 0x76, 0xB0};
    static const uint8_t quality_data[8] = {0x44, 0x22, 0xF8, 0xFF, 0x97, 0x7E, 0xD5, 0x9C};
    drive_can_frame_t frame;
    drive_boot_epoch_t boot = {0x89ABCDEFu};
    drive_boot_epoch_t decoded_boot;
    drive_session_t session = {0x89ABCDEFu, 0x1234u};
    drive_session_t decoded_session;
    drive_speed_command_t speed = {0x1234u, 0x7Eu, 1200, 10u};
    drive_speed_command_t decoded_speed;
    drive_speed_command_t reverse = {0xBEEFu, 0x00u, -1500, 5u};
    drive_state_t state = {0x2244u, -1234, -425};
    drive_state_t decoded_state;
    drive_quality_t quality = {
        0x2244u, 0xFFF8u,
        DRIVE_QUALITY_HALL_VALID | DRIVE_QUALITY_RPM_VALID | DRIVE_QUALITY_IQ_VALID |
            DRIVE_QUALITY_LOW_SPEED_OR_REVERSAL_DEGRADED |
            DRIVE_QUALITY_APPLIED_COMMAND_VALID,
        0x7Eu
    };
    drive_quality_t decoded_quality;
    drive_feedback_pair_t pair;

    CHECK(drive_encode_boot_epoch(DRIVE_NODE_LEFT, &boot, &frame) == DRIVE_OK);
    CHECK(frame_is(&frame, 0x04D1u, boot_data));
    CHECK(drive_crc16_ccitt_false(0x04D1u, 8u, boot_data, 6u) == 0xD15Fu);
    CHECK(drive_decode_boot_epoch(&frame, DRIVE_NODE_LEFT, &decoded_boot) == DRIVE_OK);
    CHECK(decoded_boot.boot_epoch == boot.boot_epoch);

    CHECK(drive_encode_session_begin(DRIVE_NODE_LEFT, &session, &frame) == DRIVE_OK);
    CHECK(frame_is(&frame, 0x04E1u, begin_data));
    CHECK(drive_decode_session_begin(&frame, DRIVE_NODE_LEFT, &decoded_session) == DRIVE_OK);
    CHECK(decoded_session.boot_epoch == session.boot_epoch &&
          decoded_session.session_id == session.session_id);
    CHECK(drive_encode_session_close(DRIVE_NODE_LEFT, &session, &frame) == DRIVE_OK);
    CHECK(frame_is(&frame, 0x04F1u, close_data));
    CHECK(drive_decode_session_close(&frame, DRIVE_NODE_LEFT, &decoded_session) == DRIVE_OK);

    CHECK(drive_encode_speed_command(DRIVE_NODE_LEFT, &speed, &frame) == DRIVE_OK);
    CHECK(frame_is(&frame, 0x0501u, speed_left_data));
    CHECK(drive_decode_speed_command(&frame, DRIVE_NODE_LEFT, &decoded_speed) == DRIVE_OK);
    CHECK(decoded_speed.target_motor_rpm == 1200 && decoded_speed.sequence == 0x7Eu);
    CHECK(drive_encode_speed_command(DRIVE_NODE_RIGHT, &reverse, &frame) == DRIVE_OK);
    CHECK(frame_is(&frame, 0x0502u, speed_right_data));
    CHECK(drive_decode_speed_command(&frame, DRIVE_NODE_RIGHT, &decoded_speed) == DRIVE_OK);
    CHECK(decoded_speed.target_motor_rpm == -1500 && decoded_speed.session_id == 0xBEEFu);

    CHECK(drive_encode_state(DRIVE_NODE_LEFT, &state, &frame) == DRIVE_OK);
    CHECK(frame_is(&frame, 0x0521u, state_data));
    CHECK(drive_decode_state(&frame, DRIVE_NODE_LEFT, &decoded_state) == DRIVE_OK);
    CHECK(decoded_state.sample_sequence == state.sample_sequence &&
          decoded_state.mechanical_rpm == state.mechanical_rpm &&
          decoded_state.iq_centiamp == state.iq_centiamp);
    CHECK(drive_encode_quality(DRIVE_NODE_LEFT, &quality, &frame) == DRIVE_OK);
    CHECK(frame_is(&frame, 0x0541u, quality_data));
    CHECK(drive_decode_quality(&frame, DRIVE_NODE_LEFT, &decoded_quality) == DRIVE_OK);
    CHECK(decoded_quality.sample_sequence == quality.sample_sequence &&
          decoded_quality.flags == quality.flags);
    CHECK(drive_pair_feedback(&state, &quality, &pair) == DRIVE_OK);
}

static void test_codec_rejects_bad_frames(void)
{
    drive_can_frame_t frame;
    drive_boot_epoch_t boot = {0x01020304u};
    drive_boot_epoch_t decoded_boot;
    drive_session_t session = {0x01020304u, 0x0001u};
    drive_session_t decoded_session;
    drive_speed_command_t speed = {0x0001u, 4u, 500, 10u};
    drive_speed_command_t decoded_speed;

    CHECK(drive_encode_boot_epoch(DRIVE_NODE_LEFT, &boot, &frame) == DRIVE_OK);
    CHECK(drive_decode_boot_epoch(&frame, DRIVE_NODE_LEFT, &decoded_boot) == DRIVE_OK);
    CHECK(decoded_boot.boot_epoch == boot.boot_epoch);

    frame.data[7] ^= 0x01u;
    CHECK(drive_decode_boot_epoch(&frame, DRIVE_NODE_LEFT, &decoded_boot) == DRIVE_ERR_CRC);
    frame.data[7] ^= 0x01u;
    frame.data[4] = 1u;
    frame_refresh_crc(&frame);
    CHECK(drive_decode_boot_epoch(&frame, DRIVE_NODE_LEFT, &decoded_boot) == DRIVE_ERR_RESERVED);
    frame.data[4] = 0u;
    frame_refresh_crc(&frame);
    CHECK(drive_decode_boot_epoch(&frame, DRIVE_NODE_RIGHT, &decoded_boot) == DRIVE_ERR_FRAME);
    frame.extended = true;
    CHECK(drive_decode_boot_epoch(&frame, DRIVE_NODE_LEFT, &decoded_boot) == DRIVE_ERR_FRAME);
    frame.extended = false;
    frame.remote = true;
    CHECK(drive_decode_boot_epoch(&frame, DRIVE_NODE_LEFT, &decoded_boot) == DRIVE_ERR_FRAME);
    frame.remote = false;
    frame.dlc = 7u;
    CHECK(drive_decode_boot_epoch(&frame, DRIVE_NODE_LEFT, &decoded_boot) == DRIVE_ERR_FRAME);

    CHECK(drive_encode_session_begin(DRIVE_NODE_LEFT, &session, &frame) == DRIVE_OK);
    CHECK(drive_decode_session_begin(&frame, DRIVE_NODE_LEFT, &decoded_session) == DRIVE_OK);
    CHECK(decoded_session.boot_epoch == session.boot_epoch &&
          decoded_session.session_id == session.session_id);
    frame.data[4] = 0u;
    frame.data[5] = 0u;
    frame.data[6] = 0u;
    frame.data[7] = 0u;
    CHECK(drive_decode_session_begin(&frame, DRIVE_NODE_LEFT, &decoded_session) == DRIVE_ERR_CRC);

    CHECK(drive_encode_speed_command(DRIVE_NODE_LEFT, &speed, &frame) == DRIVE_OK);
    CHECK(drive_decode_speed_command(&frame, DRIVE_NODE_LEFT, &decoded_speed) == DRIVE_OK);
    CHECK(decoded_speed.target_motor_rpm == 500 && decoded_speed.sequence == 4u);
    CHECK(drive_encode_speed_command(DRIVE_NODE_LEFT,
          &(drive_speed_command_t){0x0001u, 1u, INT16_MIN, 1u}, &frame) == DRIVE_ERR_RANGE);
    CHECK(drive_encode_speed_command(DRIVE_NODE_LEFT,
          &(drive_speed_command_t){0x0001u, 1u, 100, 0u}, &frame) == DRIVE_ERR_RANGE);
    CHECK(drive_encode_speed_command(DRIVE_NODE_LEFT,
          &(drive_speed_command_t){0x0001u, 1u, 100, 11u}, &frame) == DRIVE_ERR_RANGE);

    CHECK(drive_encode_boot_epoch(DRIVE_NODE_LEFT,
          &(drive_boot_epoch_t){0u}, &frame) == DRIVE_ERR_EPOCH);
    CHECK(drive_encode_session_begin(DRIVE_NODE_LEFT,
          &(drive_session_t){0u, 1u}, &frame) == DRIVE_ERR_RANGE);
    CHECK(drive_encode_session_begin(DRIVE_NODE_LEFT,
          &(drive_session_t){1u, 0u}, &frame) == DRIVE_ERR_RANGE);
    CHECK(drive_encode_speed_command((drive_node_t)0,
          &speed, &frame) == DRIVE_ERR_NODE);
}

static void test_session_sequence_ttl_and_restart(void)
{
    const uint32_t epoch = 0xAABBCCDDu;
    drive_receiver_t receiver;
    drive_session_t session = {epoch, 1u};
    drive_session_t other = {epoch, 2u};
    drive_session_t wrong_epoch = {epoch + 1u, 3u};
    drive_speed_command_t command = {1u, 255u, 1200, 10u};
    drive_speed_command_t got;

    drive_receiver_init(&receiver, DRIVE_NODE_LEFT, epoch);
    CHECK(drive_receiver_begin(&receiver, &wrong_epoch) == DRIVE_ERR_EPOCH);
    CHECK(drive_receiver_accept_speed(&receiver, &command, 1000u) == DRIVE_ERR_SESSION);
    CHECK(drive_receiver_begin(&receiver, &session) == DRIVE_OK);
    CHECK(drive_receiver_begin(&receiver, &session) == DRIVE_OK);
    CHECK(drive_receiver_accept_speed(&receiver, &command, 1000u) == DRIVE_OK);
    CHECK(drive_receiver_begin(&receiver, &other) == DRIVE_ERR_SESSION);

    CHECK(drive_receiver_poll(&receiver, 1099u, &got));
    CHECK(got.target_motor_rpm == 1200);
    CHECK(!drive_receiver_poll(&receiver, 1100u, &got));
    CHECK(drive_receiver_accept_speed(&receiver, &command, 1100u) == DRIVE_ERR_SEQUENCE);

    command.sequence = 0u;
    CHECK(drive_receiver_accept_speed(&receiver, &command, 1101u) == DRIVE_OK);
    CHECK(drive_receiver_begin(&receiver, &session) == DRIVE_OK);
    CHECK(drive_receiver_accept_speed(&receiver, &command, 1102u) == DRIVE_ERR_SEQUENCE);
    command.sequence = 2u;
    CHECK(drive_receiver_accept_speed(&receiver, &command, 1103u) == DRIVE_OK);
    command.sequence = 1u;
    CHECK(drive_receiver_accept_speed(&receiver, &command, 1104u) == DRIVE_ERR_SEQUENCE);

    CHECK(drive_receiver_close(&receiver, &other) == DRIVE_ERR_SESSION);
    CHECK(drive_receiver_close(&receiver, &session) == DRIVE_OK);
    CHECK(drive_receiver_close(&receiver, &session) == DRIVE_OK);
    CHECK(!drive_receiver_poll(&receiver, 1105u, &got));
    CHECK(drive_receiver_begin(&receiver, &session) == DRIVE_ERR_SESSION);
    CHECK(drive_receiver_begin(&receiver, &other) == DRIVE_OK);
    command.session_id = 1u;
    command.sequence = 3u;
    CHECK(drive_receiver_accept_speed(&receiver, &command, 1106u) == DRIVE_ERR_SESSION);
    command.session_id = 2u;
    CHECK(drive_receiver_accept_speed(&receiver, &command, 1107u) == DRIVE_OK);

    drive_receiver_init(&receiver, DRIVE_NODE_LEFT, epoch + 1u);
    CHECK(drive_receiver_begin(&receiver, &session) == DRIVE_ERR_EPOCH);
    wrong_epoch.boot_epoch = epoch + 1u;
    wrong_epoch.session_id = 1u;
    CHECK(drive_receiver_begin(&receiver, &wrong_epoch) == DRIVE_OK);
}

static void test_ttl_wrap_and_feedback_pair(void)
{
    drive_receiver_t receiver;
    drive_session_t session = {7u, 1u};
    drive_speed_command_t command = {1u, 0u, -300, 1u};
    drive_state_t state = {0x2244u, -1234, -425};
    drive_quality_t quality = {
        0x2244u, 0xFFF8u,
        DRIVE_QUALITY_HALL_VALID | DRIVE_QUALITY_RPM_VALID | DRIVE_QUALITY_IQ_VALID |
            DRIVE_QUALITY_LOW_SPEED_OR_REVERSAL_DEGRADED |
            DRIVE_QUALITY_APPLIED_COMMAND_VALID,
        0xFFu
    };
    drive_feedback_pair_t pair;

    drive_receiver_init(&receiver, DRIVE_NODE_RIGHT, 7u);
    CHECK(drive_receiver_begin(&receiver, &session) == DRIVE_OK);
    CHECK(drive_receiver_accept_speed(&receiver, &command, UINT32_MAX - 4u) == DRIVE_OK);
    CHECK(drive_receiver_poll(&receiver, 4u, NULL));
    CHECK(!drive_receiver_poll(&receiver, 5u, NULL));

    CHECK(drive_pair_feedback(&state, &quality, &pair) == DRIVE_OK);
    CHECK(pair.state.mechanical_rpm == -1234 && pair.state.iq_centiamp == -425);
    CHECK((pair.quality.flags & DRIVE_QUALITY_IQ_VALID) != 0u);
    CHECK((pair.quality.flags & DRIVE_QUALITY_LOW_SPEED_OR_REVERSAL_DEGRADED) != 0u);
    CHECK(pair.quality.applied_command_sequence == 0xFFu);

    quality.sample_sequence++;
    CHECK(drive_pair_feedback(&state, &quality, &pair) == DRIVE_ERR_PAIR);
    quality.sample_sequence--;
    quality.flags &= (uint8_t)~DRIVE_QUALITY_RPM_VALID;
    CHECK(drive_pair_feedback(&state, &quality, &pair) == DRIVE_ERR_RANGE);
    state.mechanical_rpm = INT16_MIN;
    CHECK(drive_pair_feedback(&state, &quality, &pair) == DRIVE_OK);
    quality.flags &= (uint8_t)~DRIVE_QUALITY_IQ_VALID;
    state.iq_centiamp = INT16_MIN;
    CHECK(drive_pair_feedback(&state, &quality, &pair) == DRIVE_OK);
    quality.flags &= (uint8_t)~DRIVE_QUALITY_APPLIED_COMMAND_VALID;
    CHECK(drive_pair_feedback(&state, &quality, &pair) == DRIVE_OK);
    quality.applied_command_sequence = 0x01u;
    CHECK(drive_pair_feedback(&state, &quality, &pair) == DRIVE_ERR_RANGE);

    CHECK(drive_feedback_is_fresh(1000u, 900u, DRIVE_FEEDBACK_STALE_LIMIT_MS));
    CHECK(!drive_feedback_is_fresh(1001u, 900u, DRIVE_FEEDBACK_STALE_LIMIT_MS));
    CHECK(drive_feedback_is_fresh(4u, UINT32_MAX - 95u, DRIVE_FEEDBACK_STALE_LIMIT_MS));
    CHECK(!drive_feedback_is_fresh(5u, UINT32_MAX - 95u, DRIVE_FEEDBACK_STALE_LIMIT_MS));
}

int main(void)
{
    test_golden_vectors();
    test_codec_rejects_bad_frames();
    test_session_sequence_ttl_and_restart();
    test_ttl_wrap_and_feedback_pair();
    printf("checks=%u failures=%u\n", checks, failures);
    return failures == 0u ? 0 : 1;
}
