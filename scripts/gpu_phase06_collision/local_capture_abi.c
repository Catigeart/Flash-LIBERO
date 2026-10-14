#include <stddef.h>
#include <mujoco/mujoco.h>
void p06c_local_contact_abi(size_t* out) {
  out[0] = sizeof(mjtNum); out[1] = sizeof(int);
  out[2] = sizeof(mjContact); out[3] = _Alignof(mjContact);
#define P06C_OFFSET(k, field) out[k] = offsetof(mjContact, field)
  P06C_OFFSET(4, dist); P06C_OFFSET(5, pos); P06C_OFFSET(6, frame);
  P06C_OFFSET(7, includemargin); P06C_OFFSET(8, friction);
  P06C_OFFSET(9, solref); P06C_OFFSET(10, solreffriction);
  P06C_OFFSET(11, solimp); P06C_OFFSET(12, mu); P06C_OFFSET(13, H);
  P06C_OFFSET(14, dim); P06C_OFFSET(15, geom1); P06C_OFFSET(16, geom2);
  P06C_OFFSET(17, exclude); P06C_OFFSET(18, efc_address);
}

unsigned int p06c_local_mxcsr(void) {
  unsigned int value;
  __asm__ volatile("stmxcsr %0" : "=m"(value));
  return value;
}
