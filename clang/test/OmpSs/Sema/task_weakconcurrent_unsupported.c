// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

#pragma oss task depend(weak, inoutset: *p) weakconcurrent(*p) // expected-error{{dependency type 'inoutset' cannot be combined with others}} expected-error{{unexpected OmpSs-2 clause 'weakconcurrent' in directive '#pragma oss task'}}
void foo1(int *p);

void bar(int *p) {
    #pragma oss task depend(weak, inoutset: *p) weakconcurrent(*p) // expected-error{{dependency type 'inoutset' cannot be combined with others}} expected-error{{unexpected OmpSs-2 clause 'weakconcurrent' in directive '#pragma oss task'}}
    {}
}
