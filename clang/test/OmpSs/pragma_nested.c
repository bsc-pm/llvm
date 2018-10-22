// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

#define t _Pragma("oss taskwait")

int main(void) {
    #pragma oss taskwait t // expected-error {{unexpected OmpSs directive}}
}
