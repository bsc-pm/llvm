; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'multideps.ll'
source_filename = "multideps.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._depend_unpack_t = type { [10 x i32]*, i64, i64, i64, i64, i64, i64 }
%struct.ASDF = type { i32, i32, i32, i32, i32, i32 }

@v = global [10 x [10 x i32]] zeroinitializer, align 16

; NOTE: this test is made by hand

; int v[10][10];
; void foo(int lb1, int ub1, int step1, int lb2, int ub2, int step2) {
;     int i, j;
;     // #pragma oss task in( { v[i][j], i=lb1:ub1:step1, j=lb2:ub2:step2 } ) 
;     #pragma oss task in( v[i][j] ) 
;     { }
; }
; 
; int main() {
;     foo(0, 9, 1, 0, 9, 1);
; } 

; Function Attrs: noinline nounwind optnone
define void @foo(i32 %lb1, i32 %ub1, i32 %step1, i32 %lb2, i32 %ub2, i32 %step2) #0 !dbg !6 {
entry:
  %lb1.addr = alloca i32, align 4
  %ub1.addr = alloca i32, align 4
  %step1.addr = alloca i32, align 4
  %lb2.addr = alloca i32, align 4
  %ub2.addr = alloca i32, align 4
  %step2.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 %lb1, i32* %lb1.addr, align 4
  store i32 %ub1, i32* %ub1.addr, align 4
  store i32 %step1, i32* %step1.addr, align 4
  store i32 %lb2, i32* %lb2.addr, align 4
  store i32 %ub2, i32* %ub2.addr, align 4
  store i32 %step2, i32* %step2.addr, align 4
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"),
    "QUAL.OSS.SHARED"([10 x [10 x i32]]* @v),
    "QUAL.OSS.FIRSTPRIVATE"(i32* %i),
    "QUAL.OSS.FIRSTPRIVATE"(i32* %j),
    "QUAL.OSS.FIRSTPRIVATE"(i32* %lb1.addr),
    "QUAL.OSS.FIRSTPRIVATE"(i32* %ub1.addr),
    "QUAL.OSS.FIRSTPRIVATE"(i32* %step1.addr),
    "QUAL.OSS.FIRSTPRIVATE"(i32* %lb2.addr),
    "QUAL.OSS.FIRSTPRIVATE"(i32* %ub2.addr),
    "QUAL.OSS.FIRSTPRIVATE"(i32* %step2.addr),
    "QUAL.OSS.MULTIDEP.RANGE.IN"(
    i32* %i, i32* %j,
    %struct.ASDF (i32*, i32*, i32*, i32*, i32*, i32*)* @compute_multi, i32* %lb1.addr, i32* %ub1.addr, i32* %step1.addr, i32* %lb2.addr, i32* %ub2.addr, i32* %step2.addr, 
    [10 x [10 x i32]]* @v,
    %struct._depend_unpack_t (i32*, i32*, [10 x [10 x i32]]*)* @compute_dep, i32* %i, i32* %j, [10 x [10 x i32]]* @v
) ], !dbg !9
  call void @llvm.directive.region.exit(token %0), !dbg !10
  ret void, !dbg !11
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j, [10 x [10 x i32]]* %v) {
entry:
  %return.val = alloca %struct._depend_unpack_t, align 8
  %0 = load i32, i32* %j, align 4, !dbg !9
  %1 = sext i32 %0 to i64
  %2 = add i64 %1, 1
  %3 = load i32, i32* %i, align 4, !dbg !9
  %4 = sext i32 %3 to i64
  %5 = add i64 %4, 1
  %arraydecay = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* %v, i64 0, i64 0, !dbg !9
  %6 = mul i64 %1, 4
  %7 = mul i64 %2, 4
  %8 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 0
  store [10 x i32]* %arraydecay, [10 x i32]** %8, align 8
  %9 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 1
  store i64 40, i64* %9, align 8
  %10 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 2
  store i64 %6, i64* %10, align 8
  %11 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 3
  store i64 %7, i64* %11, align 8
  %12 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 4
  store i64 10, i64* %12, align 8
  %13 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 5
  store i64 %4, i64* %13, align 8
  %14 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 6
  store i64 %5, i64* %14, align 8
  %15 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, align 8
  ret %struct._depend_unpack_t %15
}

define %struct.ASDF @compute_multi(i32* %lb1.addr, i32* %ub1.addr, i32* %step1.addr, i32* %lb2.addr, i32* %ub2.addr, i32* %step2.addr) #0 {
entry:
  %retval = alloca %struct.ASDF, align 4
  %retval_load = load %struct.ASDF, %struct.ASDF* %retval, align 4

  %lb1 = load i32, i32* %lb1.addr, align 4
  %ub1 = load i32, i32* %ub1.addr, align 4
  %step1 = load i32, i32* %step1.addr, align 4

  %lb2 = load i32, i32* %lb2.addr, align 4
  %ub2 = load i32, i32* %ub2.addr, align 4
  %step2 = load i32, i32* %step2.addr, align 4

  %.fca.0.insert = insertvalue %struct.ASDF %retval_load, i32 %lb1, 0
  %.fca.1.insert = insertvalue %struct.ASDF %.fca.0.insert, i32 %ub1, 1
  %.fca.2.insert = insertvalue %struct.ASDF %.fca.1.insert, i32 %step1, 2
  %.fca.3.insert = insertvalue %struct.ASDF %.fca.2.insert, i32 %lb2, 3
  %.fca.4.insert = insertvalue %struct.ASDF %.fca.3.insert, i32 %ub2, 4
  %.fca.5.insert = insertvalue %struct.ASDF %.fca.4.insert, i32 %step2, 5
  ret %struct.ASDF %.fca.5.insert
}

; CHECK: define internal void @nanos6_unpacked_deps_foo0([10 x [10 x i32]]* %v, i32* %i, i32* %j, i32* %lb1.addr, i32* %ub1.addr, i32* %step1.addr, i32* %lb2.addr, i32* %ub2.addr, i32* %step2.addr, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK: entry:
; CHECK-NEXT:   br label %0
; CHECK: 0:                                                ; preds = %entry
; CHECK-NEXT:   %1 = call %struct.ASDF @compute_multi(i32* %lb1.addr, i32* %ub1.addr, i32* %step1.addr, i32* %lb2.addr, i32* %ub2.addr, i32* %step2.addr)
; CHECK-NEXT:   %2 = extractvalue %struct.ASDF %1, 0
; CHECK-NEXT:   store i32 %2, i32* %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK: for.cond:                                         ; preds = %for.incr, %0
; CHECK-NEXT:   %3 = call %struct.ASDF @compute_multi(i32* %lb1.addr, i32* %ub1.addr, i32* %step1.addr, i32* %lb2.addr, i32* %ub2.addr, i32* %step2.addr)
; CHECK-NEXT:   %4 = extractvalue %struct.ASDF %3, 1
; CHECK-NEXT:   %5 = load i32, i32* %i, align 4
; CHECK-NEXT:   %6 = icmp sle i32 %5, %4
; CHECK-NEXT:   br i1 %6, label %for.body, label %23
; CHECK: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   %7 = call %struct.ASDF @compute_multi(i32* %lb1.addr, i32* %ub1.addr, i32* %step1.addr, i32* %lb2.addr, i32* %ub2.addr, i32* %step2.addr)
; CHECK-NEXT:   %8 = extractvalue %struct.ASDF %7, 3
; CHECK-NEXT:   store i32 %8, i32* %j, align 4
; CHECK-NEXT:   br label %for.cond1
; CHECK: for.cond1:                                        ; preds = %for.incr3, %for.body
; CHECK-NEXT:   %9 = call %struct.ASDF @compute_multi(i32* %lb1.addr, i32* %ub1.addr, i32* %step1.addr, i32* %lb2.addr, i32* %ub2.addr, i32* %step2.addr)
; CHECK-NEXT:   %10 = extractvalue %struct.ASDF %9, 4
; CHECK-NEXT:   %11 = load i32, i32* %j, align 4
; CHECK-NEXT:   %12 = icmp sle i32 %11, %10
; CHECK-NEXT:   br i1 %12, label %for.body2, label %for.incr
; CHECK: for.body2:                                        ; preds = %for.cond1
; CHECK-NEXT:   %13 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j, [10 x [10 x i32]]* %v)
; CHECK-NEXT:   %14 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j, [10 x [10 x i32]]* %v)
; CHECK-NEXT:   %15 = extractvalue %struct._depend_unpack_t %13, 0
; CHECK-NEXT:   %16 = bitcast [10 x i32]* %15 to i8*
; CHECK-NEXT:   %17 = extractvalue %struct._depend_unpack_t %13, 1
; CHECK-NEXT:   %18 = extractvalue %struct._depend_unpack_t %13, 2
; CHECK-NEXT:   %19 = extractvalue %struct._depend_unpack_t %14, 3
; CHECK-NEXT:   %20 = extractvalue %struct._depend_unpack_t %13, 4
; CHECK-NEXT:   %21 = extractvalue %struct._depend_unpack_t %13, 5
; CHECK-NEXT:   %22 = extractvalue %struct._depend_unpack_t %14, 6
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo2(i8* %handler, i32 0, i8* null, i8* %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22)
; CHECK-NEXT:   br label %for.incr3
; CHECK: 23:                                               ; preds = %for.cond
; CHECK-NEXT:   ret void
; CHECK: for.incr:                                         ; preds = %for.cond1
; CHECK-NEXT:   %24 = call %struct.ASDF @compute_multi(i32* %lb1.addr, i32* %ub1.addr, i32* %step1.addr, i32* %lb2.addr, i32* %ub2.addr, i32* %step2.addr)
; CHECK-NEXT:   %25 = extractvalue %struct.ASDF %24, 2
; CHECK-NEXT:   %26 = load i32, i32* %i, align 4
; CHECK-NEXT:   %27 = add i32 %26, %25
; CHECK-NEXT:   store i32 %27, i32* %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK: for.incr3:                                        ; preds = %for.body2
; CHECK-NEXT:   %28 = call %struct.ASDF @compute_multi(i32* %lb1.addr, i32* %ub1.addr, i32* %step1.addr, i32* %lb2.addr, i32* %ub2.addr, i32* %step2.addr)
; CHECK-NEXT:   %29 = extractvalue %struct.ASDF %28, 5
; CHECK-NEXT:   %30 = load i32, i32* %j, align 4
; CHECK-NEXT:   %31 = add i32 %30, %29
; CHECK-NEXT:   store i32 %31, i32* %j, align 4
; CHECK-NEXT:   br label %for.cond1
; CHECK-NEXT: }

; Function Attrs: noinline nounwind optnone
define i32 @main() #0 !dbg !12 {
entry:
  call void @foo(i32 0, i32 9, i32 1, i32 0, i32 9, i32 1), !dbg !13
  ret i32 0, !dbg !14
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "multideps.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 5, scope: !6)
!10 = !DILocation(line: 6, scope: !6)
!11 = !DILocation(line: 7, scope: !6)
!12 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 9, type: !8, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!13 = !DILocation(line: 10, scope: !12)
!14 = !DILocation(line: 11, scope: !12)
