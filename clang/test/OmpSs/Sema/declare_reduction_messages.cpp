// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -std=c++98 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -std=c++11 %s -Wuninitialized





int temp; // expected-note 7 {{'temp' declared here}}

#pragma oss declare reduction                                              // expected-error {{expected '(' after 'declare reduction'}}
#pragma oss declare reduction {                                            // expected-error {{expected '(' after 'declare reduction'}}
#pragma oss declare reduction(                                             // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}}
#pragma oss declare reduction(#                                            // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}}
#pragma oss declare reduction(/                                            // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}}
#pragma oss declare reduction(+                                            // expected-error {{expected ':'}}
#pragma oss declare reduction(operator                                     // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}}
#pragma oss declare reduction(operator:                                    // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}} expected-error {{expected a type}}
#pragma oss declare reduction(oper:                                        // expected-error {{expected a type}}
#pragma oss declare reduction(oper;                                        // expected-error {{expected ':'}} expected-error {{expected a type}}
#pragma oss declare reduction(fun : int                                    // expected-error {{expected ':'}} expected-error {{expected expression}}
#pragma oss declare reduction(+ : const int:                               // expected-error {{reduction type cannot be qualified with 'const', 'volatile' or 'restrict'}}
#pragma oss declare reduction(- : volatile int:                            // expected-error {{reduction type cannot be qualified with 'const', 'volatile' or 'restrict'}}
#pragma oss declare reduction(* : int;                                     // expected-error {{expected ','}} expected-error {{expected a type}}
#pragma oss declare reduction(& : double char:                             // expected-error {{cannot combine with previous 'double' declaration specifier}} expected-error {{expected expression}}
#pragma oss declare reduction(^ : double, char, :                          // expected-error {{expected a type}} expected-error {{expected expression}}
#pragma oss declare reduction(&& : int, S:                                 // expected-error {{unknown type name 'S'}} expected-error {{expected expression}}
#pragma oss declare reduction(|| : int, double : temp += omp_in)           // expected-error 2 {{only 'omp_in' or 'omp_out' variables are allowed in combiner expression}}
#pragma oss declare reduction(| : char, float : omp_out += ::temp)         // expected-error 2 {{only 'omp_in' or 'omp_out' variables are allowed in combiner expression}}
#pragma oss declare reduction(fun : long : omp_out += omp_in) {            // expected-warning {{extra tokens at the end of '#pragma oss declare reduction' are ignored}} expected-error {{expected 'initializer'}}
#pragma oss declare reduction(fun : unsigned : omp_out += ::temp))         // expected-warning {{extra tokens at the end of '#pragma oss declare reduction' are ignored}} expected-error {{expected 'initializer'}} expected-error {{only 'omp_in' or 'omp_out' variables are allowed in combiner expression}}
#pragma oss declare reduction(fun : long & : omp_out += omp_in)            // expected-error {{reduction type cannot be a reference type}}
#pragma oss declare reduction(fun : long(void) : omp_out += omp_in)        // expected-error {{reduction type cannot be a function type}}
#pragma oss declare reduction(fun : long[3] : omp_out += omp_in)           // expected-error {{reduction type cannot be an array type}}
#pragma oss declare reduction(fun23 : long, int, long : omp_out += omp_in) // expected-error {{redefinition of user-defined reduction for type 'long'}} expected-note {{previous definition is here}}

template <class T>
class Class1 {
 T a;
public:
  Class1() : a() {}
#pragma oss declare reduction(fun : T : temp)               // expected-error {{only 'omp_in' or 'omp_out' variables are allowed in combiner expression}}
#pragma oss declare reduction(fun1 : T : omp_out++)         // expected-note {{previous definition is here}} expected-error {{reduction type cannot be a reference type}}
#pragma oss declare reduction(fun1 : T : omp_out += omp_in) // expected-error {{redefinition of user-defined reduction for type 'T'}}
#pragma oss declare reduction(fun2 : T, T : omp_out++)      // expected-error {{reduction type cannot be a reference type}} expected-error {{redefinition of user-defined reduction for type 'T'}} expected-note {{previous definition is here}}
#pragma oss declare reduction(foo : T : omp_out += this->a) // expected-error {{invalid use of 'this' outside of a non-static member function}}
};

Class1<char &> e; // expected-note {{in instantiation of template class 'Class1<char &>' requested here}}

template <class T>
class Class2 : public Class1<T> {
#pragma oss declare reduction(fun : T : omp_out += omp_in)
};

#pragma oss declare reduction(fun222 : long : omp_out += omp_in)                                        // expected-note {{previous definition is here}}
#pragma oss declare reduction(fun222 : long : omp_out += omp_in)                                        // expected-error {{redefinition of user-defined reduction for type 'long'}}
#pragma oss declare reduction(fun1 : long : omp_out += omp_in) initializer                              // expected-error {{expected '(' after 'initializer'}}
#pragma oss declare reduction(fun2 : long : omp_out += omp_in) initializer {                            // expected-error {{expected '(' after 'initializer'}} expected-error {{expected expression}} expected-warning {{extra tokens at the end of '#pragma oss declare reduction' are ignored}}
#pragma oss declare reduction(fun3 : long : omp_out += omp_in) initializer[
#if __cplusplus <= 199711L
// expected-error@-2 {{expected '(' after 'initializer'}}
// expected-error@-3 {{expected expression}}
// expected-warning@-4 {{extra tokens at the end of '#pragma oss declare reduction' are ignored}}
#else
// expected-error@-6 {{expected '(' after 'initializer'}}
// expected-error@-7 {{expected variable name or 'this' in lambda capture list}}
// expected-error@-8 {{expected ')'}}
// expected-note@-9 {{to match this '('}}
#endif
#pragma oss declare reduction(fun4 : long : omp_out += omp_in) initializer()                            // expected-error {{expected expression}}
#pragma oss declare reduction(fun5 : long : omp_out += omp_in) initializer(temp)                        // expected-error {{only 'omp_priv' or 'omp_orig' variables are allowed in initializer expression}}
#pragma oss declare reduction(fun6 : long : omp_out += omp_in) initializer(omp_orig                     // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma oss declare reduction(fun7 : long : omp_out += omp_in) initializer(omp_priv Class1 < int > ())  // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma oss declare reduction(fun77 : long : omp_out += omp_in) initializer(omp_priv Class2 < int > ()) // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma oss declare reduction(fun8 : long : omp_out += omp_in) initializer(omp_priv 23)                 // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma oss declare reduction(fun88 : long : omp_out += omp_in) initializer(omp_priv 23))               // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{extra tokens at the end of '#pragma oss declare reduction' are ignored}}
#pragma oss declare reduction(fun9 : long : omp_out += omp_priv) initializer(omp_in = 23)               // expected-error {{use of undeclared identifier 'omp_priv'; did you mean 'omp_in'?}} expected-note {{'omp_in' declared here}}
#pragma oss declare reduction(fun10 : long : omp_out += omp_in) initializer(omp_priv = 23)

template <typename T>
T fun(T arg) {
#pragma oss declare reduction(red : T : omp_out++)
  {
#pragma oss declare reduction(red : T : omp_out++) // expected-note {{previous definition is here}}
#pragma oss declare reduction(red : T : omp_out++) // expected-error {{redefinition of user-defined reduction for type 'T'}}
#pragma oss declare reduction(fun : T : omp_out += omp_in) initializer(omp_priv = 23)
  }
  return arg;
}

template <typename T>
T foo(T arg) {
  T i;
  {
#pragma oss declare reduction(red : T : omp_out++)
#pragma oss declare reduction(red1 : T : omp_out++)   // expected-note {{previous definition is here}}
#pragma oss declare reduction(red1 : int : omp_out++) // expected-error {{redefinition of user-defined reduction for type 'int'}}
  #pragma oss task reduction (red : i)
  {
  }
  #pragma oss task reduction (red1 : i)
  {
  }
  #pragma oss task reduction (red2 : i) // expected-error {{incorrect reduction identifier, expected one of '+', '-', '*', '&', '|', '^', '&&', '||', 'min' or 'max' or declare reduction for type 'int'}}
  {
  }
  }
  {
#pragma oss declare reduction(red1 : int : omp_out++) // expected-note {{previous definition is here}}
#pragma oss declare reduction(red : T : omp_out++)
#pragma oss declare reduction(red1 : T : omp_out++) // expected-error {{redefinition of user-defined reduction for type 'int'}}
  #pragma oss task reduction (red : i)
  {
  }
  #pragma oss task reduction (red1 : i)
  {
  }
  #pragma oss task reduction (red2 : i) // expected-error {{incorrect reduction identifier, expected one of '+', '-', '*', '&', '|', '^', '&&', '||', 'min' or 'max' or declare reduction for type 'int'}}
  {
  }
  }
  return arg;
}

#pragma oss declare reduction(foo : int : ({int a = omp_in; a = a * 2; omp_out += a; }))
int main() {
  Class1<int> c1;
  int i;
  #pragma oss task reduction (::fun : c1)
  {
  }
  #pragma oss task reduction (::Class1<int>::fun : c1)
  {
  }
  #pragma oss task reduction (::Class2<int>::fun : i) // expected-error {{incorrect reduction identifier, expected one of '+', '-', '*', '&', '|', '^', '&&', '||', 'min' or 'max' or declare reduction for type 'int'}}
  {
  }
  return fun(15) + foo(15); // expected-note {{in instantiation of function template specialization 'foo<int>' requested here}}
}

#if __cplusplus == 201103L
struct A {
  A() {}
  A(const A&) = default;
};

int A_TEST() {
  A test;
#pragma oss declare reduction(+ : A : omp_out) initializer(omp_priv = A()) allocate(test) // expected-warning {{extra tokens at the end of '#pragma oss declare reduction' are ignored}}
#pragma oss task reduction(+ : test) // expected-error {{non-PODs value are not allowed in reductions}}
  {}
  return 0;
}

struct U
{
  void foo(U&, bool);
  U();
};
template <int N>
struct S
{
  int s;
  // expected-note@+1 {{'foo' declared here}}
  void foo(S &x) {};
  // expected-error@+1 {{too many arguments to function call, expected single argument 'x', have 2 arguments}}
  #pragma oss declare reduction (foo : U, S : omp_out.foo(omp_in, false))
  #pragma oss declare reduction (xxx : U, S : bar(omp_in)) // expected-error {{non-const lvalue reference to type 'S<1>' cannot bind to a value of unrelated type 'U'}}
  static void bar(S &x); // expected-note {{passing argument to parameter 'x' here}}
};
// expected-note@+1 {{in instantiation of template class 'S<1>' requested here}}
#pragma oss declare reduction (bar : S<1> : omp_out.foo(omp_in))

#endif
