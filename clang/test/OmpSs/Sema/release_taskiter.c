// RUN: %clang_cc1 -verify -fompss-2 -fompss-2=libnodes -ferror-limit 100 -o - %s

int main() {
  int x;
  #pragma oss taskiter
  for (int i = 0; i < 10; ++i)
  {
    #pragma oss release in(x) // expected-error {{'#pragma oss release' is not supported when nested with '#pragma oss taskiter'}}
    #pragma oss task
    {
      #pragma oss release in(x) // expected-error {{'#pragma oss release' is not supported when nested with '#pragma oss taskiter'}}
    }
  }
  #pragma oss taskiter
  while (x > 2)
  {
    #pragma oss release in(x) // expected-error {{'#pragma oss release' is not supported when nested with '#pragma oss taskiter'}}
    #pragma oss task
    {
      #pragma oss release in(x) // expected-error {{'#pragma oss release' is not supported when nested with '#pragma oss taskiter'}}
    }
  }
} 
