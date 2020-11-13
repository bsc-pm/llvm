// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

int main() {
    int v[10];
    int i; // expected-note {{previous declaration is here}}
    auto l = [i]() { // expected-note {{variable 'i' is explicitly captured here}}
        #pragma oss taskloop in( { v[i], i = 0; 10 } ) // expected-warning {{declaration shadows a local variable}}
        for (int j = 0; j < 10; ++j) {}
    };
}
