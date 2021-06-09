// RUN: %clang_cc1 -verify -fompss-2 -Wshadow -ferror-limit 100 -o - %s

int main() {
    int v[10];
    #pragma oss taskloop in( { v[i], i = 0; 10 } ) // expected-warning {{declaration shadows a local variable}}
    for (int i = 0; i < 10; ++i) {} // expected-note {{previous declaration is here}}
}
