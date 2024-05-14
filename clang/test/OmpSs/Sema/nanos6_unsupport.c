// RUN: %clang_cc1 -verify -fompss-2 -fompss-2=libnanos6 -ferror-limit 100 -o - %s

void foo() {
  int i;
  #pragma oss taskiter // expected-error {{'#pragma oss taskiter' is not supported in libnanos6}}
  for (i = 0; i < 100; ++i) {}
  #pragma oss taskiter // expected-error {{'#pragma oss taskiter' is not supported in libnanos6}}
  while (i < 100) {}
}
