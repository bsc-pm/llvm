// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

int main() {
    int a;
    #pragma oss taskloop collapse(3)
    for (int i = 0; i < 1; ++i) ; // expected-error {{expected 3 for loops after '#pragma oss taskloop', but found only 1}}
    #pragma oss taskloop collapse(3)
    for (int i = 0; i < 2; ++i) {
        a = 4; // expected-error {{not enough perfectly nested loops}}
        a = 6;
        for (int j = 0; j < 10; ++j) ;
    }
    #pragma oss taskloop collapse(3)
    for (int i = 0; i < 3; ++i) { // expected-error {{expected 3 for loops after '#pragma oss taskloop', but found only 2}}
      for (int j = 0; j < 10; ++j) {
            a = 4;
        }
    }
    #pragma oss taskloop collapse(3)
    for (int i = 0; i < 4; ++i) { // expected-error {{expected 3 for loops after '#pragma oss taskloop', but found only 1}}
        #pragma oss taskloop
        for (int j = 0; j < 10; ++j) ;
    }
    #pragma oss taskloop collapse(1)
    for (int = ; i < 5; ++i) { // expected-error {{expected identifier or '('}} expected-error {{expected ';' in 'for' statement specifier}} expected-error 2 {{use of undeclared identifier 'i'}} expected-error {{initialization clause of OmpSs-2 for loop is not in canonical form ('var = init' or 'T var = init')}}
        #pragma oss taskloop
        for (int j = 0; j < 10; ++j) ;
    }
}


