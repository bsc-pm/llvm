// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

int main(void) {
  int a;
  #pragma oss taskwait if(0) // expected-error {{unexpected OmpSs-2 clause 'if' in directive '#pragma oss taskwait'}}
}

