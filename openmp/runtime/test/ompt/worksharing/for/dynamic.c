// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base.h
// REQUIRES: ompt
// clang-format on
// UNSUPPORTED: ompv-gomp

#define SCHEDULE dynamic
#include "base.h"
