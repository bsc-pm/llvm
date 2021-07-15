// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s
int array[10];
int main() {
    #pragma oss taskloop collapse(2) in( array[i + j] )
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 1; ++j) {
        }
    }
    #pragma oss taskloop collapse(2) in( array[i + j] )// expected-error 2 {{unsupported dependencies over iterators of a non-rectangular loop}}
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < i; ++j) {
        }
    }
    #pragma oss taskloop collapse(2) in( { array[i + j], k=0;10 } ) // expected-error 2 {{unsupported dependencies over iterators of a non-rectangular loop}}
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < i; ++j) {
        }
    }
}
