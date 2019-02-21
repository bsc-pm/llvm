// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

struct S {
  int x;
};

int array[5];

int main() {
  S s;
  S *ps;

  int i;
  int *p;
  #pragma oss task private(i) depend(inout: i) // expected-error {{the data-sharing 'private' conflicts with 'shared' required by the dependency}}
  #pragma oss task depend(inout: i) private(i) // expected-error {{the data-sharing 'private' conflicts with 'shared' required by the dependency}}
  #pragma oss task firstprivate(i) depend(inout: i) // expected-error {{the data-sharing 'firstprivate' conflicts with 'shared' required by the dependency}}
  #pragma oss task shared(p) depend(inout: p[2]) // expected-error {{the data-sharing 'shared' conflicts with 'firstprivate' required by the dependency}}
  #pragma oss task shared(p) depend(inout: *p) // expected-error {{the data-sharing 'shared' conflicts with 'firstprivate' required by the dependency}}
  #pragma oss task firstprivate(array) depend(inout: array[2]) // expected-error {{the data-sharing 'firstprivate' conflicts with 'shared' required by the dependency}}
  #pragma oss task firstprivate(s) depend(inout: s.x) // expected-error {{the data-sharing 'firstprivate' conflicts with 'shared' required by the dependency}}
  #pragma oss task shared(ps) depend(inout: ps->x) // expected-error {{the data-sharing 'shared' conflicts with 'firstprivate' required by the dependency}}
  {}
}

void foo() {
    int array[10];
    int a[10];
    int i, j;
    #pragma oss task depend(in: array[a[i]], a[i]) // expected-error {{the data-sharing 'firstprivate' conflicts with 'shared' required by the dependency}}
    {}
    #pragma oss task depend(in: array[i+j]) shared(i) // expected-error {{the data-sharing 'shared' conflicts with 'firstprivate' required by the dependency}}
    {}
}

