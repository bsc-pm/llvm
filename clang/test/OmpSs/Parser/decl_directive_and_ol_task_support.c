// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

#pragma oss whatever // expected-error {{declarative directives and outlined tasks are not supported yet}}

struct S {
    #pragma oss whatever // expected-error {{declarative directives and outlined tasks are not supported yet}}
};

int main() {
}
