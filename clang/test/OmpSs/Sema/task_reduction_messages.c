// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 150 -o - %s

int incomplete[];

void test() {
#pragma oss task reduction(+ : incomplete) // expected-error {{expression has incomplete type 'int[]'}}
  ;
}

void foo() {
    int x;
    int v[10];
    int *p;
    #pragma oss task in(v[x]) reduction(+: x) in(x) reduction(+: x) // expected-error 3 {{variable 'x' conflicts between dependency and reduction clause}}
    {}
    #pragma oss task reduction(+: x) in(v[x]) reduction(+: x) in(v[x]) // expected-error 3 {{variable 'x' conflicts between dependency and reduction clause}}
    {}
    #pragma oss task reduction(+: [x]v) in(x) // This is fine since 'x' is weak restriction
    {}
    #pragma oss task in(x) reduction(+: [x]v) // This is fine since 'x' is weak restriction
    {}
    #pragma oss task reduction(+: [x]v) reduction(+: [x]p) // This is fine since 'x' is weak restriction
    {}
}

// complete to suppress an additional warning, but it's too late for pragmas
int incomplete[3];
