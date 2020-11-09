; RUN: opt -ompss-2-regions -analyze -disable-checks -print-verbosity=reduction_inits_combiners < %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

; int main() {
;     int x, y;
;     #pragma oss declare reduction(asdf : int : omp_out) initializer(omp_priv = 0)
;     #pragma oss task reduction(+ : x, y)
;     {}
;     #pragma oss task reduction(+ : x, y)
;     {}
;     #pragma oss task reduction(asdf : x, y)
;     {}
;     #pragma oss task reduction(asdf : x, y)
;     {}
; }

define i32 @main() {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %x), "QUAL.OSS.SHARED"(i32* %y), "QUAL.OSS.DEP.REDUCTION"(i32 6000, i32* %x, [2 x i8] c"x\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %x), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %x, void (i32*, i32*, i64)* @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %x, void (i32*, i32*, i64)* @red_comb), "QUAL.OSS.DEP.REDUCTION"(i32 6000, i32* %y, [2 x i8] c"y\00", %struct._depend_unpack_t.0 (i32*)* @compute_dep.1, i32* %y), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %y, void (i32*, i32*, i64)* @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %y, void (i32*, i32*, i64)* @red_comb) ]
  call void @llvm.directive.region.exit(token %0)
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %x), "QUAL.OSS.SHARED"(i32* %y), "QUAL.OSS.DEP.REDUCTION"(i32 6000, i32* %x, [2 x i8] c"x\00", %struct._depend_unpack_t.1 (i32*)* @compute_dep.2, i32* %x), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %x, void (i32*, i32*, i64)* @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %x, void (i32*, i32*, i64)* @red_comb), "QUAL.OSS.DEP.REDUCTION"(i32 6000, i32* %y, [2 x i8] c"y\00", %struct._depend_unpack_t.2 (i32*)* @compute_dep.3, i32* %y), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %y, void (i32*, i32*, i64)* @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %y, void (i32*, i32*, i64)* @red_comb) ]
  call void @llvm.directive.region.exit(token %1)
  %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %x), "QUAL.OSS.SHARED"(i32* %y), "QUAL.OSS.DEP.REDUCTION"(i32 -1, i32* %x, [2 x i8] c"x\00", %struct._depend_unpack_t.3 (i32*)* @compute_dep.6, i32* %x), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %x, void (i32*, i32*, i64)* @red_init.4), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %x, void (i32*, i32*, i64)* @red_comb.5), "QUAL.OSS.DEP.REDUCTION"(i32 -1, i32* %y, [2 x i8] c"y\00", %struct._depend_unpack_t.4 (i32*)* @compute_dep.7, i32* %y), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %y, void (i32*, i32*, i64)* @red_init.4), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %y, void (i32*, i32*, i64)* @red_comb.5) ]
  call void @llvm.directive.region.exit(token %2)
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %x), "QUAL.OSS.SHARED"(i32* %y), "QUAL.OSS.DEP.REDUCTION"(i32 -1, i32* %x, [2 x i8] c"x\00", %struct._depend_unpack_t.5 (i32*)* @compute_dep.8, i32* %x), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %x, void (i32*, i32*, i64)* @red_init.4), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %x, void (i32*, i32*, i64)* @red_comb.5), "QUAL.OSS.DEP.REDUCTION"(i32 -1, i32* %y, [2 x i8] c"y\00", %struct._depend_unpack_t.6 (i32*)* @compute_dep.9, i32* %y), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %y, void (i32*, i32*, i64)* @red_init.4), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %y, void (i32*, i32*, i64)* @red_comb.5) ]
  call void @llvm.directive.region.exit(token %3)
  ret i32 0
}


; CHECK: [0] TASK %0
; CHECK-NEXT:   %x @red_init @red_comb
; CHECK-NEXT:   %y @red_init @red_comb
; CHECK-NEXT: [0] TASK %1
; CHECK-NEXT:   %x @red_init @red_comb
; CHECK-NEXT:   %y @red_init @red_comb
; CHECK-NEXT: [0] TASK %2
; CHECK-NEXT:   %x @red_init.4 @red_comb.5
; CHECK-NEXT:   %y @red_init.4 @red_comb.5
; CHECK-NEXT: [0] TASK %3
; CHECK-NEXT:   %x @red_init.4 @red_comb.5
; CHECK-NEXT:   %y @red_init.4 @red_comb.5


declare token @llvm.directive.region.entry() #1
declare void @llvm.directive.region.exit(token) #1

; Function Attrs: noinline norecurse nounwind
declare void @red_init(i32* %0, i32* %1, i64 %2)
declare void @red_comb(i32* %0, i32* %1, i64 %2)
declare void @red_init.4(i32* %0, i32* %1, i64 %2)
declare void @red_comb.5(i32* %0, i32* %1, i64 %2)

%struct._depend_unpack_t = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.0 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.1 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.2 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.3 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.4 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.5 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.6 = type { i32*, i64, i64, i64 }

declare %struct._depend_unpack_t @compute_dep(i32* %x)
declare %struct._depend_unpack_t.0 @compute_dep.1(i32* %y)
declare %struct._depend_unpack_t.1 @compute_dep.2(i32* %x)
declare %struct._depend_unpack_t.2 @compute_dep.3(i32* %y)
declare %struct._depend_unpack_t.3 @compute_dep.6(i32* %x)
declare %struct._depend_unpack_t.4 @compute_dep.7(i32* %y)
declare %struct._depend_unpack_t.5 @compute_dep.8(i32* %x)
declare %struct._depend_unpack_t.6 @compute_dep.9(i32* %y)
