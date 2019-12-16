// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s
// expected-no-diagnostics
template<typename T> T foo() { return T(); }

void bar(int n) {
    int vla[n];
    #pragma oss task cost(foo<int>())
    {}
    #pragma oss task cost(n)
    {}
    #pragma oss task cost(vla[1])
    {}
}

// CHECK: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.COST"(i32 %call) ]
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %n.addr), "QUAL.OSS.COST"(i32 %4) ]
// CHECK: %7 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1), "QUAL.OSS.COST"(i32 %6) ]
