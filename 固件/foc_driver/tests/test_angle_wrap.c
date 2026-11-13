/* Exercise the actual private function without linking the hardware-dependent code. */
#include <stdio.h>
#include <stdlib.h>
#include "../BusinessLayer/Src/bl_foc.c"

/* Link-only stand-ins: these functions are never used by the angle test. */
BOOL bsp_hall_read(U8 *code) { (void)code; return FALSE; }
F32 dl_sense_get_bus_v(VOID) { return 0.0F; }
S32 dl_sense_get_phase_current_ma(DL_SENSE_PHASE_E phase) { (void)phase; return 0; }
VOID dl_motor_set_duty(F32 u, F32 v, F32 w) { (void)u; (void)v; (void)w; }

int main(int argc, char **argv)
{
    float input;
    if (argc != 2) return 2;
    input = strtof(argv[1], NULL);
    printf("%a\n", (double)_bl_foc_wrap_angle(input));
    return 0;
}
