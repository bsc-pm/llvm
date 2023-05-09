// RUN: %clang_cc1 -verify -fompss-2 -fompss-fpga-hls-tasks-dir %{fs-tmp-root} -fompss-fpga -fompss-fpga-wrapper-code -fompss-fpga-dump -ferror-limit 100 -o - %s
void dependencyNormalFunc();  // expected-error {{This function is depended by an fpga kernel, but we couldn't locate its body. Make sure it is visible in this translation unit.}}

#pragma oss task device(fpga)
void missingDependBody() { // expected-note {{The fpga kernel}}
    dependencyNormalFunc();
}