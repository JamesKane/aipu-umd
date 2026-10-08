/* SPDX-License-Identifier: BSD-2-Clause */
/* For the driver's UAPI header in FreeBSD user space. */
#include <sys/types.h>
#include <stdint.h>
typedef uint8_t __u8;
typedef uint16_t __u16;
typedef uint32_t __u32;
typedef unsigned long long __u64;	/* As Linux's (int-ll64.h). */
typedef int8_t __s8;
typedef int16_t __s16;
typedef int32_t __s32;
typedef long long __s64;
#define	__user
