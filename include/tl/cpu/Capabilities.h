//
// Created by renyz on 2026/3/21.
//

#ifndef CTORCH_CAPABILITIES_H
#define CTORCH_CAPABILITIES_H

#include "Features.h"

#if defined(ARCH_X86_FAMILY)
#if defined(HAS_AVX512F) && defined(HAS_AVX512CD) && defined(HAS_AVX512BW) && defined(HAS_AVX512DQ)
  #define HAS_CPU_CAPABILITY_AVX512 1
#endif
#if defined(HAS_AVX2)
  #define HAS_CPU_CAPABILITY_AVX2 1
#endif
#if defined(HAS_AVX)
  #define HAS_CPU_CAPABILITY_AVX 1
#endif
#endif // ARCH_X86_FAMILY
#if defined(ARCH_ARM_FAMILY)
  #warning "TODO support ARM NEON & SVE"
#endif // ARCH_ARM_FAMILY

#if defined(CPU_CAPABILITY_AVX512) || (defined(HAS_CPU_CAPABILITY_AVX512) && !defined(CPU_CAPABILITY))
  #if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_AVX512)
    #error "CPU capability redefined or does not supported by compiler option"
  #endif
  #define CPU_CAPABILITY AVX512
  #define CPU_CAPABILITY_AVX512 1
  #define VEC_WIDTH 512
#endif // CPU_CAPABILITY_AVX512

#if defined(CPU_CAPABILITY_AVX2) || (defined(HAS_CPU_CAPABILITY_AVX2) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_AVX2)
    #error "CPU capability redefined or does not supported by compiler option"
  #endif
  #define CPU_CAPABILITY AVX2
  #define CPU_CAPABILITY_AVX2 1
  #define VEC_WIDTH 256
#endif // CPU_CAPABILITY_AVX2

#if defined(CPU_CAPABILITY_AVX) || (defined(HAS_CPU_CAPABILITY_AVX) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_AVX)
    #error "CPU capability redefined or does not supported by compiler option"
  #endif
  #define CPU_CAPABILITY AVX
  #define CPU_CAPABILITY_AVX
  #define VEC_WIDTH 128
#endif // CPU_CAPABILITY_AVX

#if defined(CPU_CAPABILITY_NEON) || (defined(HAS_CPU_CAPABILITY_NEON) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_NEON)
    #error "CPU capability redefined or does not supported by compiler option"
  #endif
  #define CPU_CAPABILITY NEON
  #define CPU_CAPABILITY_NEON 1
  #define VEC_WIDTH 128
#endif // CPU_CAPABILITY_NEON

#if defined(CPU_CAPABILITY_SVE) || (defined(HAS_CPU_CAPABILITY_SVE) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_SVE)
    #error "CPU capability redefined or does not supported by compiler option"
  #endif
  #define CPU_CAPABILITY SVE
  #define CPU_CAPABILITY_SVE 1
  #define VEC_WIDTH (-1) // scalable
#endif // CPU_CAPABILITY_SVE

#if !defined(CPU_CAPABILITY)
  #define CPU_CAPABILITY GENERIC
  #define CPU_CAPABILITY_GENERIC 1
  #define VEC_WIDTH 128 // default width for scalar vector implementation
#endif // CPU_CAPABILITY

#endif //CTORCH_CAPABILITIES_H
