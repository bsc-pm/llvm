// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int main()
{
  int a = 0;
  #pragma oss task in(a)
  {
    #pragma oss task reduction(+: a)
    a++;
  }
  return 0;
}

// The red_init/red_comb allocas should not be between these instructions
// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %a), "QUAL.OSS.DEP.IN"(i32* %a, [2 x i8] c"a\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %a) ]
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %a), "QUAL.OSS.DEP.REDUCTION"(i32 6000, i32* %a, [2 x i8] c"a\00", %struct._depend_unpack_t.0 (i32*)* @compute_dep.1, i32* %a), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %a, void (i32*, i32*, i64)* @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %a, void (i32*, i32*, i64)* @red_comb) ]
