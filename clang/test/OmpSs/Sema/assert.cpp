// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

#pragma oss assert // expected-error {{expected '(' after 'assert'}}
// expected-error@+1 {{expected string literal}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma oss assert(
// expected-error@+1 {{expected string literal}}
#pragma oss assert()
// expected-warning@+1 {{missing terminating '"' character}} expected-error@+1 {{expected string literal}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma oss assert("
// expected-warning@+1 {{missing terminating '"' character}} expected-error@+1 {{expected string literal}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma oss assert(")
#pragma oss assert("")
#pragma oss assert("") asdf // expected-warning {{extra tokens at the end of '#pragma oss assert' are ignored}}


namespace A {
  #pragma oss assert("")
  namespace B {
    #pragma oss assert("")
  }
}

template <typename T> T foo() {
  #pragma oss assert("") // expected-error {{'#pragma oss assert' directive must appear only in file scope}}
}

class C {
  #pragma oss assert("") // expected-error {{'#pragma oss assert' directive must appear only in file scope}}
};

int main() {
  #pragma oss assert("") // expected-error {{'#pragma oss assert' directive must appear only in file scope}}
}
