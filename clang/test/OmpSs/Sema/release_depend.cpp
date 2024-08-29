// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

void foo() {
}

struct S1; // expected-note {{declared here}}

class vector {
  public:
    int operator[](int index) { return 0; }
};

struct S2 {
        static int s2;
};

int S2::s2 = 3;

int main(int argc, char **argv, char *env[]) {
  vector vec;
  typedef float V __attribute__((vector_size(16)));
  V a;
  auto arr = x; // expected-error {{use of undeclared identifier 'x'}}

  argv[0 : 5] = 1; // expected-error {{OmpSs-2 array section is not allowed here}}
  argv[0 ; 5] = 1; // expected-error {{OmpSs-2 array section is not allowed here}}

  int array[10][20];

  #pragma oss release depend // expected-error {{expected '(' after 'depend'}}
  #pragma oss release depend ( // expected-error {{expected 'in', 'out', 'inout', 'inoutset' or 'mutexinoutset' in OmpSs-2 clause 'depend'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma oss release depend () // expected-error {{expected 'in', 'out', 'inout', 'inoutset' or 'mutexinoutset' in OmpSs-2 clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma oss release depend (argc // expected-error {{expected 'in', 'out', 'inout', 'inoutset' or 'mutexinoutset' in OmpSs-2 clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss release depend (in : argc)) // expected-warning {{extra tokens at the end of '#pragma oss release' are ignored}}
  #pragma oss release depend (out: ) // expected-error {{expected expression}}
  #pragma oss release depend (out :S1) // expected-error {{'S1' does not refer to a value}}
  #pragma oss release depend(in : argv[1][1] = '2')
  #pragma oss release depend (in : vec[1]) // expected-error {{expected addressable lvalue expression, array element, array shape or array section}}
  #pragma oss release depend (in : argv[0])
  #pragma oss release depend (in : ) // expected-error {{expected expression}}
  #pragma oss release depend (in : main) // expected-error {{expected addressable lvalue expression, array element, array shape or array section}}
  #pragma oss release depend(in : a[0]) // expected-error{{expected addressable lvalue expression, array element, array shape or array section}}
  #pragma oss release depend (in : argv[ // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma oss release depend (in : argv[: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma oss release depend (in : argv[:] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss release depend (in : argv[:] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss release depend (in : argv[argc: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma oss release depend (in : argv[argc:argc] // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss release depend (in : argv[0:-1]) // expected-error {{section length is evaluated to a negative value -1}}
  #pragma oss release depend (in : argv[-1:0])
  #pragma oss release depend (in : argv[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma oss release depend (in : argv[3:4:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  #pragma oss release depend(in:a[0:1]) // expected-error {{subscripted value is not an array or pointer}}
  #pragma oss release depend(in:argv[argv[:2]:1]) // expected-error {{OmpSs-2 array section is not allowed here}}
  #pragma oss release depend(in:argv[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma oss release depend(in:env[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}}
  #pragma oss release depend(in : argv[ : argc][1 : argc - 1]) // expected-error {{pointer types only allow one-level array sections}}
  #pragma oss release depend(in : array[1 ; 2][3 : 4]) // expected-error {{array section form is not valid in 'depend' clause}} 
  #pragma oss release depend(in : array[1 : 2][3 ; 4]) // expected-error {{array section form is not valid in 'depend' clause}} 
  #pragma oss release depend(in : array[1 ; 2][3 ; 4]) // expected-error {{array section form is not valid in 'depend' clause}} 

  #pragma oss release in(argv[; // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma oss release in(argv[;] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss release in(argv[;] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss release in(argv[argc; // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma oss release in(argv[argc;argc] // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss release in(argv[0;-1]) // expected-error {{section length is evaluated to a negative value -1}}
  #pragma oss release in(argv[-1;0])
  #pragma oss release in(argv[;]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma oss release in(argv[3;4;1]) // expected-error {{expected ']'}} expected-note {{to match this '['}} expected-error {{expected ',' or ')' in 'in' clause}} expected-error {{expected expression}}
  #pragma oss release in(a[0;1]) // expected-error {{subscripted value is not an array or pointer}}
  #pragma oss release in(argv[argv[;2]:1]) // expected-error {{OmpSs-2 array section is not allowed here}}
  #pragma oss release in(argv[0;][;]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma oss release in(env[0;][;]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}}
  #pragma oss release in( argv[ ; argc][1 ; argc - 1]) // expected-error {{pointer types only allow one-level array sections}}
  #pragma oss release mutexinoutset(argc) // expected-warning {{extra tokens at the end of '#pragma oss release' are ignored}}
  #pragma oss release inoutset(argc) // expected-warning {{extra tokens at the end of '#pragma oss release' are ignored}}

  #pragma oss release depend(in : arr[0])
  #pragma oss release depend(, // expected-error {{expected 'in', 'out', 'inout', 'inoutset', 'mutexinoutset' or 'weak' in OmpSs-2 clause 'depend'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma oss release depend(, : // expected-error {{expected 'in', 'out', 'inout', 'inoutset', 'mutexinoutset' or 'weak' in OmpSs-2 clause 'depend'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma oss release depend(, : ) // expected-error {{expected 'in', 'out', 'inout', 'inoutset', 'mutexinoutset' or 'weak' in OmpSs-2 clause 'depend'}}
  #pragma oss release depend(in, weak: ) // expected-error {{expected expression}}
  #pragma oss release depend(weak, in: ) // expected-error {{expected expression}}
  #pragma oss release depend(, weak: argc) // expected-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OmpSs-2 clause 'depend'}}
  #pragma oss release depend(weak, : argc) // expected-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OmpSs-2 clause 'depend'}}
  #pragma oss release depend(weak, weak: argc) // expected-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OmpSs-2 clause 'depend'}}
  #pragma oss release depend(in, in: argc) // expected-error {{expected 'weak' dependency type}}
  #pragma oss release depend(out, in: argc) // expected-error {{expected 'weak' dependency type}}
  #pragma oss release depend(weak, in: 1) // expected-error {{expected addressable lvalue expression, array element, array shape or array section}}
  #pragma oss release depend(weak, in: S2::s2)
  #pragma oss release depend(weak, mutexinoutset: argc)
  #pragma oss release depend(kk, inoutset: argc) // expected-error {{dependency type 'inoutset' cannot be combined with others}}
  #pragma oss release depend(kk: argc) // expected-error {{expected 'in', 'out', 'inout', 'inoutset' or 'mutexinoutset' in OmpSs-2 clause 'depend'}}

  return 0;
}
