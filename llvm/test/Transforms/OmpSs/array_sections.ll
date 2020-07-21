; RUN: opt %s -ompss-2 -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; int mat[7][3];
; int main() {
;     #pragma oss task in(mat[:][:])
;     {}
;     #pragma oss task in(mat[0:1][1:1])
;     {}
; }

%struct._depend_unpack_t = type { [3 x i32]*, i64, i64, i64, i64, i64, i64 }
%struct._depend_unpack_t.0 = type { [3 x i32]*, i64, i64, i64, i64, i64, i64 }

@mat = common dso_local global [7 x [3 x i32]] zeroinitializer, align 16

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !6 {
entry:
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([7 x [3 x i32]]* @mat), "QUAL.OSS.DEP.IN"([7 x [3 x i32]]* @mat, %struct._depend_unpack_t ([7 x [3 x i32]]*)* @compute_dep, [7 x [3 x i32]]* @mat) ], !dbg !8
  call void @llvm.directive.region.exit(token %0), !dbg !9
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([7 x [3 x i32]]* @mat), "QUAL.OSS.DEP.IN"([7 x [3 x i32]]* @mat, %struct._depend_unpack_t.0 ([7 x [3 x i32]]*)* @compute_dep.1, [7 x [3 x i32]]* @mat) ], !dbg !10
  call void @llvm.directive.region.exit(token %1), !dbg !11
  ret i32 0, !dbg !12
}

define internal %struct._depend_unpack_t @compute_dep([7 x [3 x i32]]* %mat) {
entry:
  %return.val = alloca %struct._depend_unpack_t, align 8
  %arraydecay = getelementptr inbounds [7 x [3 x i32]], [7 x [3 x i32]]* %mat, i64 0, i64 0, !dbg !13
  %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 0
  store [3 x i32]* %arraydecay, [3 x i32]** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 1
  store i64 12, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 3
  store i64 12, i64* %3, align 8
  %4 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 4
  store i64 7, i64* %4, align 8
  %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 5
  store i64 0, i64* %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 6
  store i64 7, i64* %6, align 8
  %7 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, align 8
  ret %struct._depend_unpack_t %7
}

define internal %struct._depend_unpack_t.0 @compute_dep.1([7 x [3 x i32]]* %mat) {
entry:
  %return.val = alloca %struct._depend_unpack_t.0, align 8
  %arraydecay = getelementptr inbounds [7 x [3 x i32]], [7 x [3 x i32]]* %mat, i64 0, i64 0, !dbg !14
  %0 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 0
  store [3 x i32]* %arraydecay, [3 x i32]** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 1
  store i64 12, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 2
  store i64 4, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 3
  store i64 8, i64* %3, align 8
  %4 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 4
  store i64 7, i64* %4, align 8
  %5 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 5
  store i64 0, i64* %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 6
  store i64 2, i64* %6, align 8
  %7 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, align 8
  ret %struct._depend_unpack_t.0 %7
}

; CHECK: define internal void @nanos6_unpacked_deps_main0([7 x [3 x i32]]* %mat, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call %struct._depend_unpack_t @compute_dep([7 x [3 x i32]]* %mat)
; CHECK-NEXT:   %1 = call %struct._depend_unpack_t @compute_dep([7 x [3 x i32]]* %mat)
; CHECK-NEXT:   %2 = extractvalue %struct._depend_unpack_t %0, 0
; CHECK-NEXT:   %3 = bitcast [3 x i32]* %2 to i8*
; CHECK-NEXT:   %4 = extractvalue %struct._depend_unpack_t %0, 1
; CHECK-NEXT:   %5 = extractvalue %struct._depend_unpack_t %0, 2
; CHECK-NEXT:   %6 = extractvalue %struct._depend_unpack_t %1, 3
; CHECK-NEXT:   %7 = extractvalue %struct._depend_unpack_t %0, 4
; CHECK-NEXT:   %8 = extractvalue %struct._depend_unpack_t %0, 5
; CHECK-NEXT:   %9 = extractvalue %struct._depend_unpack_t %1, 6
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo2(i8* %handler, i32 0, i8* null, i8* %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_deps_main0(%nanos6_task_args_main0* %task_args, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_mat = getelementptr %nanos6_task_args_main0, %nanos6_task_args_main0* %task_args, i32 0, i32 0
; CHECK-NEXT:   %load_gep_mat = load [7 x [3 x i32]]*, [7 x [3 x i32]]** %gep_mat
; CHECK-NEXT:   call void @nanos6_unpacked_deps_main0([7 x [3 x i32]]* %load_gep_mat, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_deps_main1([7 x [3 x i32]]* %mat, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call %struct._depend_unpack_t.0 @compute_dep.1([7 x [3 x i32]]* %mat)
; CHECK-NEXT:   %1 = call %struct._depend_unpack_t.0 @compute_dep.1([7 x [3 x i32]]* %mat)
; CHECK-NEXT:   %2 = extractvalue %struct._depend_unpack_t.0 %0, 0
; CHECK-NEXT:   %3 = bitcast [3 x i32]* %2 to i8*
; CHECK-NEXT:   %4 = extractvalue %struct._depend_unpack_t.0 %0, 1
; CHECK-NEXT:   %5 = extractvalue %struct._depend_unpack_t.0 %0, 2
; CHECK-NEXT:   %6 = extractvalue %struct._depend_unpack_t.0 %1, 3
; CHECK-NEXT:   %7 = extractvalue %struct._depend_unpack_t.0 %0, 4
; CHECK-NEXT:   %8 = extractvalue %struct._depend_unpack_t.0 %0, 5
; CHECK-NEXT:   %9 = extractvalue %struct._depend_unpack_t.0 %1, 6
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo2(i8* %handler, i32 0, i8* null, i8* %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_deps_main1(%nanos6_task_args_main1* %task_args, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_mat = getelementptr %nanos6_task_args_main1, %nanos6_task_args_main1* %task_args, i32 0, i32 0
; CHECK-NEXT:   %load_gep_mat = load [7 x [3 x i32]]*, [7 x [3 x i32]]** %gep_mat
; CHECK-NEXT:   call void @nanos6_unpacked_deps_main1([7 x [3 x i32]]* %load_gep_mat, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1


attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "array_sections.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 3, column: 13, scope: !6)
!9 = !DILocation(line: 4, column: 6, scope: !6)
!10 = !DILocation(line: 5, column: 13, scope: !6)
!11 = !DILocation(line: 6, column: 6, scope: !6)
!12 = !DILocation(line: 7, column: 1, scope: !6)
!13 = !DILocation(line: 3, column: 25, scope: !6)
!14 = !DILocation(line: 5, column: 25, scope: !6)
