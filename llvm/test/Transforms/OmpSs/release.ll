; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'release.ll'
source_filename = "release.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; int main() {
;     int x;
;     int array[10][10];
;     #pragma oss release in(x, array)
;     #pragma oss task in(x, array)
;     {
;         #pragma oss release in(x, array)
;     }
; }

%struct._depend_unpack_t = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.0 = type { [10 x [10 x i32]]*, i64, i64, i64, i64, i64, i64 }
%struct._depend_unpack_t.1 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.2 = type { [10 x [10 x i32]]*, i64, i64, i64, i64, i64, i64 }
%struct._depend_unpack_t.3 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.4 = type { [10 x [10 x i32]]*, i64, i64, i64, i64, i64, i64 }

; Function Attrs: noinline nounwind optnone
define i32 @main() #0 !dbg !6 {
entry:
  %x = alloca i32, align 4
  %array = alloca [10 x [10 x i32]], align 16
  %0 = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00"), "QUAL.OSS.DEP.IN"(i32* %x, [2 x i8] c"x\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %x), "QUAL.OSS.DEP.IN"([10 x [10 x i32]]* %array, [6 x i8] c"array\00", %struct._depend_unpack_t.0 ([10 x [10 x i32]]*)* @compute_dep.1, [10 x [10 x i32]]* %array) ], !dbg !9
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %x), "QUAL.OSS.SHARED"([10 x [10 x i32]]* %array), "QUAL.OSS.DEP.IN"(i32* %x, [2 x i8] c"x\00", %struct._depend_unpack_t.1 (i32*)* @compute_dep.2, i32* %x), "QUAL.OSS.DEP.IN"([10 x [10 x i32]]* %array, [6 x i8] c"array\00", %struct._depend_unpack_t.2 ([10 x [10 x i32]]*)* @compute_dep.3, [10 x [10 x i32]]* %array) ], !dbg !10
  %2 = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00"), "QUAL.OSS.DEP.IN"(i32* %x, [2 x i8] c"x\00", %struct._depend_unpack_t.3 (i32*)* @compute_dep.4, i32* %x), "QUAL.OSS.DEP.IN"([10 x [10 x i32]]* %array, [6 x i8] c"array\00", %struct._depend_unpack_t.4 ([10 x [10 x i32]]*)* @compute_dep.5, [10 x [10 x i32]]* %array) ], !dbg !11
  call void @llvm.directive.region.exit(token %1), !dbg !12
  ret i32 0, !dbg !13
; CHECK-LABEL: @main(
; CHECK: %0 = call %struct._depend_unpack_t @compute_dep(i32* %x)
; CHECK-NEXT: %1 = extractvalue %struct._depend_unpack_t %0, 0
; CHECK-NEXT: %2 = bitcast i32* %1 to i8*
; CHECK-NEXT: %3 = extractvalue %struct._depend_unpack_t %0, 1
; CHECK-NEXT: %4 = extractvalue %struct._depend_unpack_t %0, 2
; CHECK-NEXT: %5 = extractvalue %struct._depend_unpack_t %0, 3
; CHECK-NEXT: call void @nanos6_release_read_1(i8* %2, i64 %3, i64 %4, i64 %5)
; CHECK-NEXT: %6 = call %struct._depend_unpack_t.0 @compute_dep.1([10 x [10 x i32]]* %array)
; CHECK-NEXT: %7 = extractvalue %struct._depend_unpack_t.0 %6, 0
; CHECK-NEXT: %8 = bitcast [10 x [10 x i32]]* %7 to i8*
; CHECK-NEXT: %9 = extractvalue %struct._depend_unpack_t.0 %6, 1
; CHECK-NEXT: %10 = extractvalue %struct._depend_unpack_t.0 %6, 2
; CHECK-NEXT: %11 = extractvalue %struct._depend_unpack_t.0 %6, 3
; CHECK-NEXT: %12 = extractvalue %struct._depend_unpack_t.0 %6, 4
; CHECK-NEXT: %13 = extractvalue %struct._depend_unpack_t.0 %6, 5
; CHECK-NEXT: %14 = extractvalue %struct._depend_unpack_t.0 %6, 6
; CHECK-NEXT: call void @nanos6_release_read_2(i8* %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14)
}

; CHECK-LABEL: @nanos6_unpacked_task_region_main0(
; CHECK: %1 = call %struct._depend_unpack_t.3 @compute_dep.4(i32* %x)
; CHECK-NEXT: %2 = extractvalue %struct._depend_unpack_t.3 %1, 0
; CHECK-NEXT: %3 = bitcast i32* %2 to i8*
; CHECK-NEXT: %4 = extractvalue %struct._depend_unpack_t.3 %1, 1
; CHECK-NEXT: %5 = extractvalue %struct._depend_unpack_t.3 %1, 2
; CHECK-NEXT: %6 = extractvalue %struct._depend_unpack_t.3 %1, 3
; CHECK-NEXT: call void @nanos6_release_read_1(i8* %3, i64 %4, i64 %5, i64 %6)
; CHECK-NEXT: %7 = call %struct._depend_unpack_t.4 @compute_dep.5([10 x [10 x i32]]* %array)
; CHECK-NEXT: %8 = extractvalue %struct._depend_unpack_t.4 %7, 0
; CHECK-NEXT: %9 = bitcast [10 x [10 x i32]]* %8 to i8*
; CHECK-NEXT: %10 = extractvalue %struct._depend_unpack_t.4 %7, 1
; CHECK-NEXT: %11 = extractvalue %struct._depend_unpack_t.4 %7, 2
; CHECK-NEXT: %12 = extractvalue %struct._depend_unpack_t.4 %7, 3
; CHECK-NEXT: %13 = extractvalue %struct._depend_unpack_t.4 %7, 4
; CHECK-NEXT: %14 = extractvalue %struct._depend_unpack_t.4 %7, 5
; CHECK-NEXT: %15 = extractvalue %struct._depend_unpack_t.4 %7, 6
; CHECK-NEXT: call void @nanos6_release_read_2(i8* %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15)

define internal %struct._depend_unpack_t @compute_dep(i32* %x) #1 !dbg !14 {
entry:
  %retval = alloca %struct._depend_unpack_t, align 8
  %x.addr = alloca i32*, align 8
  store i32* %x, i32** %x.addr, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
  store i32* %x, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
  store i64 4, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
  store i64 4, i64* %3, align 8
  %4 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
  ret %struct._depend_unpack_t %4
}

define internal %struct._depend_unpack_t.0 @compute_dep.1([10 x [10 x i32]]* %array) #1 !dbg !15 {
entry:
  %retval = alloca %struct._depend_unpack_t.0, align 8
  %array.addr = alloca [10 x [10 x i32]]*, align 8
  store [10 x [10 x i32]]* %array, [10 x [10 x i32]]** %array.addr, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
  store [10 x [10 x i32]]* %array, [10 x [10 x i32]]** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
  store i64 40, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
  store i64 40, i64* %3, align 8
  %4 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 4
  store i64 10, i64* %4, align 8
  %5 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 5
  store i64 0, i64* %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 6
  store i64 10, i64* %6, align 8
  %7 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8
  ret %struct._depend_unpack_t.0 %7
}

; Function Attrs: nounwind
declare i1 @llvm.directive.marker() #2

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #2

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #2

define internal %struct._depend_unpack_t.1 @compute_dep.2(i32* %x) #1 !dbg !16 {
entry:
  %retval = alloca %struct._depend_unpack_t.1, align 8
  %x.addr = alloca i32*, align 8
  store i32* %x, i32** %x.addr, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 0
  store i32* %x, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 1
  store i64 4, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 3
  store i64 4, i64* %3, align 8
  %4 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, align 8
  ret %struct._depend_unpack_t.1 %4
}

define internal %struct._depend_unpack_t.2 @compute_dep.3([10 x [10 x i32]]* %array) #1 !dbg !17 {
entry:
  %retval = alloca %struct._depend_unpack_t.2, align 8
  %array.addr = alloca [10 x [10 x i32]]*, align 8
  store [10 x [10 x i32]]* %array, [10 x [10 x i32]]** %array.addr, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 0
  store [10 x [10 x i32]]* %array, [10 x [10 x i32]]** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 1
  store i64 40, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 3
  store i64 40, i64* %3, align 8
  %4 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 4
  store i64 10, i64* %4, align 8
  %5 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 5
  store i64 0, i64* %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 6
  store i64 10, i64* %6, align 8
  %7 = load %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, align 8
  ret %struct._depend_unpack_t.2 %7
}

define internal %struct._depend_unpack_t.3 @compute_dep.4(i32* %x) #1 !dbg !18 {
entry:
  %retval = alloca %struct._depend_unpack_t.3, align 8
  %x.addr = alloca i32*, align 8
  store i32* %x, i32** %x.addr, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 0
  store i32* %x, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 1
  store i64 4, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 3
  store i64 4, i64* %3, align 8
  %4 = load %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, align 8
  ret %struct._depend_unpack_t.3 %4
}

define internal %struct._depend_unpack_t.4 @compute_dep.5([10 x [10 x i32]]* %array) #1 !dbg !19 {
entry:
  %retval = alloca %struct._depend_unpack_t.4, align 8
  %array.addr = alloca [10 x [10 x i32]]*, align 8
  store [10 x [10 x i32]]* %array, [10 x [10 x i32]]** %array.addr, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 0
  store [10 x [10 x i32]]* %array, [10 x [10 x i32]]** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 1
  store i64 40, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 3
  store i64 40, i64* %3, align 8
  %4 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 4
  store i64 10, i64* %4, align 8
  %5 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 5
  store i64 0, i64* %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 6
  store i64 10, i64* %6, align 8
  %7 = load %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, align 8
  ret %struct._depend_unpack_t.4 %7
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "min-legal-vector-width"="0" "no-jump-tables"="false" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 12.0.0"}
!6 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "release.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 4, column: 13, scope: !6)
!10 = !DILocation(line: 5, column: 13, scope: !6)
!11 = !DILocation(line: 7, column: 17, scope: !6)
!12 = !DILocation(line: 8, column: 5, scope: !6)
!13 = !DILocation(line: 9, column: 1, scope: !6)
!14 = distinct !DISubprogram(linkageName: "compute_dep", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!15 = distinct !DISubprogram(linkageName: "compute_dep.1", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = distinct !DISubprogram(linkageName: "compute_dep.2", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!17 = distinct !DISubprogram(linkageName: "compute_dep.3", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!18 = distinct !DISubprogram(linkageName: "compute_dep.4", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!19 = distinct !DISubprogram(linkageName: "compute_dep.5", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
