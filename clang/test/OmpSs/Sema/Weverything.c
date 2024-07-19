// RUN: %clang_cc1 -verify -fompss-2 %s -Weverything
// expected-no-diagnostics

// This test covers RecursiveASTVisitor
int main(void) {
  #pragma oss task
  {}
}
