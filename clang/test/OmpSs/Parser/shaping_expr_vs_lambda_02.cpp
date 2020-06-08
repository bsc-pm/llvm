// RUN: %clang_cc1 -x c++ -verify -fompss-2 -ferror-limit 100 %s
// expected-no-diagnostics

// Lambda expression declarator is optional. Check we do not
// parse these expressions as a shaping expr

void foo() {
  auto a = [&] {};
  auto b = [] {};
}
