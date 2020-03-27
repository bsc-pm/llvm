// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 150 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify -fompss-2 -std=c++98 -ferror-limit 150 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify -fompss-2 -std=c++11 -ferror-limit 150 -o - %s -Wuninitialized


void xxx(int argc) {
  int fp; // NOTE: initialize the variable 'fp' to silence this warning
#pragma oss task reduction(+:fp) // WARNING: variable 'fp' is uninitialized when used here
  for (int i = 0; i < 10; ++i)
    ;
}

void foo() {
}

void foobar(int &ref) {
#pragma oss task reduction(+:ref)
  foo();
}

struct S1; // expected-note {{declared here}} expected-note 4 {{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;
  S2 &operator+(const S2 &arg) { return (*this); }

public:
  S2() : a(0) {}
  S2(S2 &s2) : a(s2.a) {}
  static float S2s;
  static const float S2sc; // expected-note 2 {{'S2sc' declared here}}
};
const float S2::S2sc = 0;
S2 b;
const S2 ba[5];           // expected-note 2 {{'ba' defined here}}
class S3 {
  int a;

public:
  int b;
  S3() : a(0) {}
  S3(const S3 &s3) : a(s3.a) {}
  S3 operator+(const S3 &arg1) { return arg1; }
};
int operator+(const S3 &arg1, const S3 &arg2) { return 5; }
S3 c;
const S3 ca[5];     // expected-note 2 {{'ca' defined here}}
extern const int f; // expected-note 4 {{'f' declared here}}
class S4 {
  int a;
  S4();
  S4(const S4 &s4);
  S4 &operator+(const S4 &arg) { return (*this); }

public:
  S4(int v) : a(v) {}
};
S4 &operator&=(S4 &arg1, S4 &arg2) { return arg1; }
class S5 {
  int a;
  S5() : a(0) {}
  S5(const S5 &s5) : a(s5.a) {}
  S5 &operator+(const S5 &arg);

public:
  S5(int v) : a(v) {}
};
class S6 {
  int a;

public:
  S6() : a(6) {}
  operator int() { return 6; }
} o;

S3 h, k;

template <class T>       // expected-note {{declared here}}
T tmain(T argc) {
  const T d = T();       // expected-note 4 {{'d' defined here}}
  const T da[5] = {T()}; // expected-note 2 {{'da' defined here}}
  T qa[5] = {T()};
  T i, z;
  T &j = i;
  S3 &p = k;
  const T &r = da[(int)i];     // expected-note 2 {{'r' defined here}}
  T &q = qa[(int)i];
  T fl;
#pragma oss task reduction // expected-error {{expected '(' after 'reduction'}}
  foo();
#pragma oss task reduction + // expected-error {{expected '(' after 'reduction'}} expected-warning {{extra tokens at the end of '#pragma oss task' are ignored}}
  foo();
#pragma oss task reduction( // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma oss task reduction(- // expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma oss task reduction() // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}}
  foo();
#pragma oss task reduction(*) // expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected expression}}
  foo();
#pragma oss task reduction(\) // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}}
  foo();
#pragma oss task reduction(& : argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{invalid operands to binary expression ('float' and 'float')}}
  foo();
#pragma oss task reduction(| : argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{invalid operands to binary expression ('float' and 'float')}}
  foo();
#pragma oss task reduction(|| : argc ? i : argc) // expected-error 2 {{expected variable name}}
  foo();
#pragma oss task reduction(foo : argc) // expected-error {{incorrect reduction identifier, expected one of '+', '-', '*', '&', '|', '^', '&&', '||', 'min' or 'max' or declare reduction for type 'float'}} expected-error {{incorrect reduction identifier, expected one of '+', '-', '*', '&', '|', '^', '&&', '||', 'min' or 'max' or declare reduction for type 'int'}}
  foo();
#pragma oss task reduction(^ : T) // expected-error {{'T' does not refer to a value}}
  foo();
#pragma oss task reduction(+ : z, a, b, c, d, f) // expected-error {{expression has incomplete type 'S1'}} expected-error 2 {{non-PODs value are not allowed in reductions}} expected-error 3 {{const-qualified variable cannot be reduction}}
  foo();
#pragma oss task reduction(min : a, b, c, d, f) // expected-error {{expression has incomplete type 'S1'}} expected-error 2 {{non-PODs value are not allowed in reductions}} expected-error 3 {{const-qualified variable cannot be reduction}}
  foo();
#pragma oss task reduction(max : h.b) // expected-error {{expected variable name}}
  foo();
#pragma oss task reduction(+ : ba) // expected-error {{const-qualified variable cannot be reduction}}
  foo();
#pragma oss task reduction(* : ca) // expected-error {{const-qualified variable cannot be reduction}}
  foo();
#pragma oss task reduction(- : da) // expected-error {{const-qualified variable cannot be reduction}} expected-error {{const-qualified variable cannot be reduction}}
  foo();
#pragma oss task reduction(^ : fl) // expected-error {{invalid operands to binary expression ('float' and 'float')}}
  foo();
#pragma oss task reduction(&& : S2::S2s)
  foo();
#pragma oss task reduction(&& : S2::S2sc) // expected-error {{const-qualified variable cannot be reduction}}
  foo();
#pragma oss task reduction(+ : o) // expected-error {{non-PODs value are not allowed in reductions}}
  foo();
#pragma oss task reduction(+ : p), reduction(+ : p) // expected-error 2 {{non-PODs value are not allowed in reductions}}
  foo();
#pragma oss task reduction(+ : r) // expected-error 2 {{const-qualified variable cannot be reduction}}
  foo();

  return T();
}

int main(int argc, char **argv) {
  const int d = 5;       // expected-note 2 {{'d' defined here}}
  const int da[5] = {0}; // expected-note {{'da' defined here}}
  int qa[5] = {0};
  S4 e(4);
  S5 g(5);
  int i, z;
  int &j = i;
  S3 &p = k;
  const int &r = da[i];        // expected-note {{'r' defined here}}
  int &q = qa[i];
  float fl;
#pragma oss task reduction // expected-error {{expected '(' after 'reduction'}}
  foo();
#pragma oss task reduction + // expected-error {{expected '(' after 'reduction'}} expected-warning {{extra tokens at the end of '#pragma oss task' are ignored}}
  foo();
#pragma oss task reduction( // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma oss task reduction(- // expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma oss task reduction() // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}}
  foo();
#pragma oss task reduction(*) // expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected expression}}
  foo();
#pragma oss task reduction(\) // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}}
  foo();
#pragma oss task reduction(foo : argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{incorrect reduction identifier, expected one of '+', '-', '*', '&', '|', '^', '&&', '||', 'min' or 'max'}}
  foo();
#pragma oss task reduction(| : argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma oss task reduction(|| : argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  foo();
#pragma oss task reduction(~ : argc) // expected-error {{expected unqualified-id}}
  foo();
#pragma oss task reduction(&& : argc, z)
  foo();
#pragma oss task reduction(^ : S1) // expected-error {{'S1' does not refer to a value}}
  foo();
#pragma oss task reduction(+ : a, b, c, d, f) // expected-error {{expression has incomplete type 'S1'}} expected-error 2 {{const-qualified variable cannot be reduction}} expected-error 2 {{non-PODs value are not allowed in reductions}}
  foo();
#pragma oss task reduction(min : a, b, c, d, f) // expected-error {{expression has incomplete type 'S1'}} expected-error 2 {{non-PODs value are not allowed in reductions}} expected-error 2 {{const-qualified variable cannot be reduction}}
  foo();
#pragma oss task reduction(max : h.b) // expected-error {{expected variable name}}
  foo();
#pragma oss task reduction(+ : ba) // expected-error {{const-qualified variable cannot be reduction}}
  foo();
#pragma oss task reduction(* : ca) // expected-error {{const-qualified variable cannot be reduction}}
  foo();
#pragma oss task reduction(- : da) // expected-error {{const-qualified variable cannot be reduction}}
  foo();
#pragma oss task reduction(^ : fl) // expected-error {{invalid operands to binary expression ('float' and 'float')}}
  foo();
#pragma oss task reduction(&& : S2::S2s)
  foo();
#pragma oss task reduction(&& : S2::S2sc) // expected-error {{const-qualified variable cannot be reduction}}
  foo();
#pragma oss task reduction(& : e, g) // expected-error 2 {{non-PODs value are not allowed in reductions}}
  foo();
#pragma oss task reduction(+ : o) // expected-error {{non-PODs value are not allowed in reductions}}
  foo();
#pragma oss task reduction(+ : p), reduction(+ : p) // expected-error 2 {{non-PODs value are not allowed in reductions}}
  foo();
#pragma oss task reduction(+ : r) // expected-error {{const-qualified variable cannot be reduction}}
  foo();
  static int m;
#pragma oss task reduction(+ : m) // OK
  m++;

  return tmain(argc) + tmain(fl); // expected-note {{in instantiation of function template specialization 'tmain<int>' requested here}} expected-note {{in instantiation of function template specialization 'tmain<float>' requested here}}
}
