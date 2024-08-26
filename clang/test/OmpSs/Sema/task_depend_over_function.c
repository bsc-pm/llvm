// RUN: %clang_cc1 -verify -fompss-2 -x c++ -ferror-limit 100 -o - %s
// RUN: %clang_cc1 -verify -fompss-2 -x c -ferror-limit 100 -o - %s
int *get();
int main() {
    #pragma oss task in((get()[3])) // expected-error {{call expressions are not supported}}
    {}
}
