//
// Created by renyz on 2026/3/13.
//

#ifndef CTORCH_ASSERTION_H
#define CTORCH_ASSERTION_H

#include "CoreDefs.h"

#define CT_INTERNAL_RUN_WHEN_FALSE(cond, ...)\
  do {                \
    if (!(cond)) {    \
       __VA_ARGS__;   \
    }                 \
  } while (0)

/**
 * Assertion, panic when cond is false, only check in debug mode.
 * usage: CT_ASSERT(cond, message[, arg0[, arg1[, ...]]])
 *      where message and args are a format same to printf
 */
#ifdef CT_DEBUG
  #define CT_ASSERT(cond, ...) CT_INTERNAL_RUN_WHEN_FALSE(cond, ::ct::details::assertion_failed(__FILE__, __LINE__, CT_FUNC_NAME, __VA_ARGS__))
#else
  #define CT_ASSERT(cond, ...) ((void) 0)
#endif

/**
 * Runtime checks, always exists, throw std::runtime_error when check failed.
 * usage: CT_CHECK(cond, message[, arg0[, arg1[, ...]]])
 *      where message and args are a format same to printf
 */
#define CT_CHECK(cond, ...) CT_INTERNAL_RUN_WHEN_FALSE(cond, ::ct::details::check_failed(__FILE__, __LINE__, CT_FUNC_NAME, __VA_ARGS__))

namespace ct::details {

[[noreturn]] CT_NOINLINE
void assertion_failed(const char* file, int line, const char* fn_name, const char* message, ...);

[[noreturn]] CT_NOINLINE
void check_failed(const char* file, int line, const char* fn_name, const char* message, ...);

} // namespace ct::details


#endif //CTORCH_ASSERTION_H
