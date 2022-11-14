// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

int x;
int y;
#pragma oss task cost(x + 1) // expected-error {{global variables are not allowed here}}
void foo();
#pragma oss task priority(y + 1) // expected-error {{global variables are not allowed here}}
void foo1();

