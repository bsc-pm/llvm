// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

#pragma oss task depend(in, asdf: p) // expected-error {{expected 'weak' dependency type}}
void foo1(int *p) {}
#pragma oss task asdf(*a) // expected-warning {{extra tokens at the end of '#pragma oss task' are ignored}}
void foo2(int *p) {}
#pragma oss taskwait // expected-error {{unexpected OmpSs-2 directive '#pragma oss taskwait'}}
void foo3(int *p) {}
#pragma oss task // expected-error {{function declaration is expected after 'task' function directive}}
#pragma oss task
void foo4(int *p) {}

#pragma oss task shared(p) private(p) firstprivate(p) default(none) // expected-error {{unexpected OmpSs-2 clause 'shared' in directive '#pragma oss task'}} expected-error {{unexpected OmpSs-2 clause 'private' in directive '#pragma oss task'}} expected-error {{unexpected OmpSs-2 clause 'firstprivate' in directive '#pragma oss task'}} expected-error {{unexpected OmpSs-2 clause 'default' in directive '#pragma oss task'}}
void foo5(int *p) {}
#pragma oss task depend(in: p[0 ; 5]) // expected-error {{array section form is not valid in 'depend' clause}}
void foo6(int *p) {}

#pragma oss task weakconcurrent(*p) weakcommutative(*p) // expected-error {{unexpected OmpSs-2 clause 'weakconcurrent' in directive '#pragma oss task'}}
void foo7(int *p) {}
#pragma oss task reduction(+: *p) weakreduction(+: *p) // expected-error {{expected variable name or array shaping}} expected-error {{expected variable name or array shaping}} expected-error {{conflicts between dependency and reduction clause}}
void foo8(int *p) {}
#pragma oss task reduction(+: a) weakreduction(+: a) // expected-error 2 {{expected lvalue reference or array shape}} expected-error {{variable 'a' conflicts between dependency and reduction clause}}
void foo8_1(int a) {}
#pragma oss task on([1]p) // expected-error {{unexpected OmpSs-2 clause 'on' in directive '#pragma oss task'}}
void foo9(int *p) {}

struct S0 {
  // expected-error@+2 {{field 'foo' declared as a function}}
  #pragma oss task
  void foo();
};

struct S1 {
  // expected-error@+3 {{field 'bar' declared as a function}}
  // expected-error@+2 {{expected ';' at end of declaration list}}
  #pragma oss task
  void bar() {}
};

#pragma oss task // expected-error {{function declaration is expected after 'task' function directive}}
