// RUN: %clang_cc1 -verify -fompss-2 -fompss-fpga-hls-tasks-dir %{fs-tmp-root} -fompss-fpga -fompss-fpga-wrapper-code -fompss-fpga-dump -ferror-limit 100 -o - %s

#pragma oss task device(cuda) num_instances(1)  // expected-error {{num_instances cannot be used with other devices than fpga}}
void foo1();
#pragma oss task device(cuda) onto(0x300000000)  // expected-error {{onto cannot be used with other devices than fpga}}
void foo2();
#pragma oss task device(cuda) num_repetitions(1)  // expected-error {{num_repetitions cannot be used with other devices than fpga}}
void foo3();
#pragma oss task device(cuda) period(1)  // expected-error {{period cannot be used with other devices than fpga}}
void foo4();
#pragma oss task device(cuda) affinity(1)  // expected-error {{affinity cannot be used with other devices than fpga}}
void foo5();
#pragma oss task device(cuda) copy_deps  // expected-error {{copy_deps cannot be used with other devices than fpga}}
void foo6();
#pragma oss task device(cuda) copy_in([1]i)  // expected-error {{copy_in cannot be used with other devices than fpga}}
void foo7(int *i);
#pragma oss task device(cuda) copy_out([1]o)  // expected-error {{copy_out cannot be used with other devices than fpga}}
void foo8(int *o);
#pragma oss task device(cuda) copy_inout([1]io)  // expected-error {{copy_inout cannot be used with other devices than fpga}}
void foo9(int *io);
#pragma oss task device(fpga)  
void body(); // expected-error {{#pragma oss task device(fpga)' can only be applied to functions with bodies for now}}

int a;
#pragma oss task device(fpga) copy_in(a) // expected-error {{Expected this to be an array if it is intended to be placed in the local memory of the FPGA. You might want to use the copy_deps, copy_in, copy_out and/or copy_inout parameters in the oss task}}
void array0() { 
}
#pragma oss task device(fpga) copy_out([1]a) 
void array1(const int * a) { // expected-error {{Out/inout localmem with const qualifier. This parameter can't be sent to the FPGA local memory if it is marked as const and as an out/inout. Consider removing const}}
}

#pragma oss task device(fpga) copy_in([1]a) 
void array2(int b, int (*a)[b]) {  // expected-error {{Expected this to evaulate to a constant unsigned integer}}
}

void bar() {
    #pragma oss task device(fpga) // expected-error {{device(fpga) is not supported in inline directives}}
    {}
}

#pragma oss task device(fpga) onto(1) // expected-error {{Architecture bits not set in the onto clause: 33 -> SMP arch, 32 -> FPGA arch}}
void ontoerr1() {}


#pragma oss task device(fpga) onto(0x600000000) // expected-error {{Task type too wide: 2bits arch + 32bits hash}}
void ontoerr2() {}

void dependencyNormalFunc();  // expected-error {{This function is depended by an fpga kernel, but we couldn't locate its body. Make sure it is visible in this translation unit.}}

#pragma oss task device(fpga)
void missingDependBody() { // expected-note {{The fpga kernel}}
    dependencyNormalFunc();
}