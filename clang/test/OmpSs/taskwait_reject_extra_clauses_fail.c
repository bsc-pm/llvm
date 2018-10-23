// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

int main(void) {
  int a;
  #pragma oss taskwait depend(in: a) // expected-error {{unexpected OmpSs clause 'depend' in directive '#pragma oss taskwait'}}
}

