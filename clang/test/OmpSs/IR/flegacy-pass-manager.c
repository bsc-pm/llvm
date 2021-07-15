// RUN: %clang_cc1 -verify -fompss-2 -flegacy-pass-manager -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=LEGACY
// RUN: %clang_cc1 -verify -fompss-2 -fno-legacy-pass-manager -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// Weak check: match any nanos6 stuff added by the transformation phase

int main() {
  #pragma oss task
  {}
}

// LEGACY: nanos6
// CHECK: nanos6
