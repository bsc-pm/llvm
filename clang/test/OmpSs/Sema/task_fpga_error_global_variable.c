// RUN: %clang_cc1 -verify -fompss-2 -fompss-fpga-hls-tasks-dir %{fs-tmp-root} -fompss-fpga -fompss-fpga-wrapper-code -fompss-fpga-dump -ferror-limit 100 -o - %s
const int CONST_GLOBAL = 1;
#pragma oss task device(fpga) num_instances(CONST_GLOBAL)  // expected-error {{Expected this to evaulate to a constant unsigned integer}}
void foo1() {}
