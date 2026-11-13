/* Host-test stand-in only; not a replacement for the target platform type.h. */
#ifndef AGTC_HOST_TEST_TYPE_H
#define AGTC_HOST_TEST_TYPE_H
#include <stdint.h>
typedef uint8_t U8;
typedef int8_t S8;
typedef int32_t S32;
typedef float F32;
typedef int BOOL;
typedef void VOID;
typedef enum { TYPE_RET_SUCCESS, TYPE_RET_FAIL, TYPE_RET_PARAMETER_ERROR } TYPE_RET_E;
#define TRUE 1
#define FALSE 0
#define IN
#define OUT
#endif
