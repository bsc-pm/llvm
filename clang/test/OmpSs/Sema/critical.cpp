// RUN: %clang_cc1 -verify -fompss-2 %s -Wuninitialized


int foo();

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma oss critical
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

template<typename T, int N>
int tmain(int argc, char **argv) {
  #pragma oss critical
  ;
  #pragma oss critical unknown // expected-warning {{extra tokens at the end of '#pragma oss critical' are ignored}}
  #pragma oss critical ( // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss critical ( + // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss critical (name2 // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss critical (name1)
  foo();
  {
    #pragma oss critical
  } // expected-error {{expected statement}}
  return 0;
}

int main(int argc, char **argv) {
  #pragma oss critical
  ;
  #pragma oss critical unknown // expected-warning {{extra tokens at the end of '#pragma oss critical' are ignored}}
  #pragma oss critical ( // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss critical ( + // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss critical (name2 // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss critical (name1)
  foo();
  {
    #pragma oss critical
  } // expected-error {{expected statement}}
  return tmain<int, 4>(argc, argv) + tmain<float, -5>(argc, argv);
}

int foo() {
  L1: // expected-note {{jump exits scope of OmpSs-2 structured block}}
    foo();
  #pragma oss critical
  {
    foo();
    goto L1; // expected-error {{cannot jump from this goto statement to its label}}
  }
  goto L2; // expected-error {{cannot jump from this goto statement to its label}}
  #pragma oss critical
  {  // expected-note {{jump bypasses OmpSs-2 structured block}}
    L2:
    foo();
  }

  return 0;
 }
