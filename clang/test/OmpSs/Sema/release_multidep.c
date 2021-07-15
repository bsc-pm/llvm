// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

int main() {
  int x;
  int array[10];
  #pragma oss task in(array)
  {
      #pragma oss release in({ array[i], i=0;10 }) // expected-error {{expected expression}}
      {}
  }
} 
