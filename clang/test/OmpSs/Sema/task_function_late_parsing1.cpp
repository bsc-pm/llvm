// RUN: %clang_cc1 -verify -x c++ -fompss-2 -ferror-limit 100 -o - %s

// This test should emit an error similar to:
//   implicit instantiation of template 'S<int>' within its own definition
// Now it does not happer because of late parsing, but the instantiation
// of the class does not contain the task outline. The reason is because
// the instantiation occurs when building the templated task outline,
// and it has not been assigned to the FunctionDecl

// expected-error@+3 {{implicit instantiation of template 'S<int>' within its own definition}}
// expected-note@+4 {{in instantiation of template class 'S<int>' requested here}}
template<typename T>
struct S {
  #pragma oss declare reduction(asdf : T : omp_out += omp_in) initializer(omp_priv = 0)
  #pragma oss task weakreduction(S<int>::asdf: [1]p)
  void foo(int *p);
};
