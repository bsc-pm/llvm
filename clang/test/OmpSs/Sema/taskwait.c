// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

void foo(int *p) {
    #pragma oss taskwait depend(asdf, asdf: p[0]) // expected-error {{expected 'in', 'out' or 'inout' in OmpSs-2 clause 'depend'}}
    // TODO: improve diagnostic
    #pragma oss taskwait depend(weak, asdf: p[0]) // expected-error {{expected 'in', 'out' or 'inout' in OmpSs-2 clause 'depend'}}
    #pragma oss taskwait depend(asdf: p[0]) // expected-error {{expected 'in', 'out' or 'inout' in OmpSs-2 clause 'depend'}}
    #pragma oss taskwait depend(weak: p[0]) // expected-error {{expected 'in', 'out' or 'inout' in OmpSs-2 clause 'depend'}}
    #pragma oss taskwait depend(inout: p[0])
}
