// RUN: %clang_cc1 -verify -x c++ -std=c++11 -fompss-2 -ferror-limit 100 -o - %s

template<typename T>
T adder(T v) {
  return v;
}

template<typename T, typename... Args>
T adder(T first, Args... args) {
  T tmp;
  #pragma oss task firstprivate(args) in(args) // expected-error 2 {{variadic templates are not allowed in OmpSs-2 clauses}}
  {
    tmp = first + adder(args...);
  }
  return tmp;
}

int main() {
  int a = adder(1, 2, 3, 4);
}
