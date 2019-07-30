// RUN: %clang_cc1 -x c++ -verify -fompss-2 -ferror-limit 100 %s

struct S {
  int x;
  void f() {
    #pragma oss task shared(this) // expected-error {{expected variable name or data member of current class}}
    {}
  }
};

int a;
int main() {
    #pragma oss task shared(a) firstprivate(a) private(a) // expected-error {{shared variable cannot be private}} expected-error {{shared variable cannot be firstprivate}}
    #pragma oss task firstprivate(a) shared(a) private(a) // expected-error {{firstprivate variable cannot be private}} expected-error {{firstprivate variable cannot be shared}}
    #pragma oss task private(a) shared(a) firstprivate(a) // expected-error {{private variable cannot be shared}} expected-error {{private variable cannot be firstprivate}}
    { a++; }
    #pragma oss task
    {
      #pragma oss task private(a)
      {
        #pragma oss task
        { a++; }
      }
    }
    struct S s;
    #pragma oss task shared(s.x) // expected-error {{expected variable name}}
    #pragma oss task private(s.x) // expected-error {{expected variable name}}
    #pragma oss task firstprivate(s.x) // expected-error {{expected variable name}}
    { s.x++; }
}
