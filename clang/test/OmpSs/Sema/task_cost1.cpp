// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

// template<typename T> T foo() { return T(); }
void bar(int n) {
    #pragma oss task cost(foo<int>()) // expected-error {{use of undeclared identifier 'foo'}}
    {}
}
