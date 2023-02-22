// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

#pragma oss task device(asdf) // expected-error {{expected 'smp', 'cuda', 'opencl' or 'fpga' in OmpSs-2 clause 'device'}}
void foo();
#pragma oss task device(fpga) ndrange(1, 1, 1) // expected-error {{ndrange cannot be used with other devices than cuda and opencl}}
void foo1() {}

void bar() {
    #pragma oss task device(fpga) // expected-error {{device(fpga) is not supported in inline directives}}
    {}
}

