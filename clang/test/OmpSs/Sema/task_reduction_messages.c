// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 150 -o - %s

int incomplete[];

void test() {
#pragma oss task reduction(+ : incomplete) // expected-error {{expression has incomplete type 'int []'}}
  ;
}

// complete to suppress an additional warning, but it's too late for pragmas
int incomplete[3];
