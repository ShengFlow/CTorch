//
// Created by renyz on 2026/3/13.
//
#include <string>
#include <vector>
#include <cstdio>
#include <cstdarg>
#include <stdexcept>

#include "Assertion.h"

namespace ct::details {

void assertion_failed(const char* file, int line, const char* fn_name, const char* message, ...) {
  using namespace std::literals;

  va_list args;
  // Get number of bytes needed for message
  va_start(args, message);
  int needed = std::vsnprintf(nullptr, 0, message, args);
  va_end(args);

  // Format message with prepared buffer
  std::vector<char> buf(needed + 1);
  va_start(args, message);
  std::vsnprintf(buf.data(), needed + 1, message, args);
  va_end(args);

  std::string details = "\n  at "s + file + ":" + std::to_string(line) + "\n  function " + fn_name;
  std::string full = "Assertion failed: "s + buf.data() + details;
  // Throw or abort with trap
  #ifdef CT_THROW_ON_ASSERTION_FAILED
  throw std::runtime_error(full);
  #else
  std::fputs(full.c_str(), stderr);
  CT_BREAKPOINT;
  std::abort();
  #endif
}

void check_failed(const char* file, int line, const char* fn_name, const char* message, ...) {
  using namespace std::literals;

  va_list args;
  // Get number of bytes needed for message
  va_start(args, message);
  int needed = std::vsnprintf(nullptr, 0, message, args);
  va_end(args);

  // Format message with prepared buffer
  std::vector<char> buf(needed + 1);
  va_start(args, message);
  std::vsnprintf(buf.data(), needed + 1, message, args);
  va_end(args);

  std::string details = "\n  at "s + file + ":" + std::to_string(line) + "\n  function " + fn_name;
  std::string full = "Check failed: "s + buf.data() + details;
  throw std::runtime_error(full);
}

} // namespace ct::details
