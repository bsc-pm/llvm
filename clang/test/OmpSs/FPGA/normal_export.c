// RUN: %clang_cc1 -fompss-2 -fompss-fpga-hls-tasks-dir %{fs-tmp-root} -fompss-fpga -fompss-fpga-extract -fompss-fpga-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

#include "Inputs/header.h"
#include "Inputs/header.fpga.h"
const int VAL = 0;
void depend();
#pragma oss task device(fpga)
void foo() {
    #pragma HLS var=VAL
    depend();
}
void depend() {}

// CHECK:foo
// CHECK-NEXT:#include "
// CHECK-SAME:header.fpga.h"
// CHECK-NEXT:void depend();
// CHECK-NEXT:void depend() {
// CHECK-NEXT:}
// CHECK-NEXT:#pragma oss task device(fpga)
// CHECK-NEXT:void foo() {
// CHECK-NEXT:    #pragma HLS var=VAL
// CHECK-NEXT:    depend();
// CHECK-NEXT:}
