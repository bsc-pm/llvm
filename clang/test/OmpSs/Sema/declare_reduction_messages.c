// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -Wuninitialized

// UNSUPPORTED: true


int temp; // expected-note 6 {{'temp' declared here}}

#pragma oss declare reduction                                              // expected-error {{expected '(' after 'declare reduction'}}
#pragma oss declare reduction {                                            // expected-error {{expected '(' after 'declare reduction'}}
#pragma oss declare reduction(                                             // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}}
#pragma oss declare reduction(#                                            // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}}
#pragma oss declare reduction(/                                            // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}}
#pragma oss declare reduction(+                                            // expected-error {{expected ':'}}
#pragma oss declare reduction(for                                          // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}}
#pragma oss declare reduction(if:                                          // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}} expected-error {{expected a type}}
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
#pragma oss declare reduction(| : char, float : omp_out += temp)           // expected-error 2 {{only 'omp_in' or 'omp_out' variables are allowed in combiner expression}}
#pragma oss declare reduction(fun : long : omp_out += omp_in) {            // expected-error {{expected 'initializer'}} expected-warning {{extra tokens at the end of '#pragma oss declare reduction' are ignored}}
#pragma oss declare reduction(fun : unsigned : omp_out += temp))           // expected-error {{expected 'initializer'}} expected-warning {{extra tokens at the end of '#pragma oss declare reduction' are ignored}} expected-error {{only 'omp_in' or 'omp_out' variables are allowed in combiner expression}}
#pragma oss declare reduction(fun : long(void) : omp_out += omp_in)        // expected-error {{reduction type cannot be a function type}}
#pragma oss declare reduction(fun : long[3] : omp_out += omp_in)           // expected-error {{reduction type cannot be an array type}}
#pragma oss declare reduction(fun23 : long, int, long : omp_out += omp_in) // expected-error {{redefinition of user-defined reduction for type 'long'}} expected-note {{previous definition is here}}

#pragma oss declare reduction(fun222 : long : omp_out += omp_in)
#pragma oss declare reduction(fun1 : long : omp_out += omp_in) initializer                 // expected-error {{expected '(' after 'initializer'}}
#pragma oss declare reduction(fun2 : long : omp_out += omp_in) initializer {               // expected-error {{expected '(' after 'initializer'}} expected-error {{expected expression}} expected-warning {{extra tokens at the end of '#pragma oss declare reduction' are ignored}}
#pragma oss declare reduction(fun3 : long : omp_out += omp_in) initializer[                // expected-error {{expected '(' after 'initializer'}} expected-error {{expected expression}} expected-warning {{extra tokens at the end of '#pragma oss declare reduction' are ignored}}
#pragma oss declare reduction(fun4 : long : omp_out += omp_in) initializer()               // expected-error {{expected expression}}
#pragma oss declare reduction(fun5 : long : omp_out += omp_in) initializer(temp)           // expected-error {{only 'omp_priv' or 'omp_orig' variables are allowed in initializer expression}}
#pragma oss declare reduction(fun6 : long : omp_out += omp_in) initializer(omp_orig        // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma oss declare reduction(fun7 : long : omp_out += omp_in) initializer(omp_priv 12)    // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma oss declare reduction(fun8 : long : omp_out += omp_in) initializer(omp_priv = 23)  // expected-note {{previous definition is here}}
#pragma oss declare reduction(fun8 : long : omp_out += omp_in) initializer(omp_priv = 23)) // expected-warning {{extra tokens at the end of '#pragma oss declare reduction' are ignored}} expected-error {{redefinition of user-defined reduction for type 'long'}}
#pragma oss declare reduction(fun9 : long : omp_out += omp_in) initializer(omp_priv = )    // expected-error {{expected expression}}

struct S {
  int s;
};
#pragma oss declare reduction(+: struct S: omp_out.s += omp_in.s) // initializer(omp_priv = { .s = 0 })
#pragma omp declare reduction(&: struct S: omp_out.s += omp_in.s) initializer(omp_priv = { .s = 0 })
#pragma omp declare reduction(|: struct S: omp_out.s += omp_in.s) initializer(omp_priv = { 0 })

int fun(int arg) {
  struct S s;// expected-note {{'s' defined here}}
  s.s = 0;
#pragma oss task reduction(+ : s) // expected-error {{list item of type 'struct S' is not valid for specified reduction operation: unable to provide default initialization value}}
  for (arg = 0; arg < 10; ++arg)
    s.s += arg;
#pragma oss declare reduction(red : int : omp_out++)
  {
#pragma oss declare reduction(red : int : omp_out++) // expected-note {{previous definition is here}}
#pragma oss declare reduction(red : int : omp_out++) // expected-error {{redefinition of user-defined reduction for type 'int'}}
    {
#pragma oss declare reduction(red : int : omp_out++)
    }
  }
  return arg;
}
