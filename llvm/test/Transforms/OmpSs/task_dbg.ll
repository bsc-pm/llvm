; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_dbg.ll'
source_filename = "task_dbg.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Originally CodeExtractor removes intrinsics if they refer to something outside
; the region. This test checks for OmpSs-2 that debug intrinsics are kept.

; struct S {
;     int x = 4;
;     void foo() {
;         #pragma oss task
;         {
;             x++;
;             x++;
;         }
;     }
; };
;
; int main() {
;     int x = 10;
;     int vla[x];
;     int array[10];
;     S s;
;     s.foo();
;     x = vla[0] = array[0] = 43;
;     #pragma oss task
;     {
;         x++;
;         vla[0]++;
;         array[0]++;
;     }
; }

%struct.S = type { i32 }

$_ZN1SC1Ev = comdat any

$_ZN1S3fooEv = comdat any

$_ZN1SC2Ev = comdat any

; Function Attrs: mustprogress noinline norecurse nounwind optnone
define dso_local noundef i32 @main() #0 !dbg !16 {
; CHECK:    call void @llvm.dbg.declare(metadata i32* %x
; CHECK:    call void @llvm.dbg.declare(metadata i32* %vla
; CHECK:    call void @llvm.dbg.declare(metadata [10 x i32]* %array
; CHECK:       final.then:
; CHECK-NEXT:    call void @llvm.dbg.declare(metadata i32* %x
; CHECK-NEXT:    call void @llvm.dbg.declare(metadata i32* %vla
; CHECK-NEXT:    call void @llvm.dbg.declare(metadata [10 x i32]* %array
entry:
  %x = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  %array = alloca [10 x i32], align 16
  %s = alloca %struct.S, align 4
  call void @llvm.dbg.declare(metadata i32* %x, metadata !20, metadata !DIExpression()), !dbg !21
  store i32 10, i32* %x, align 4, !dbg !21
  %0 = load i32, i32* %x, align 4, !dbg !22
  %1 = zext i32 %0 to i64, !dbg !23
  %2 = call i8* @llvm.stacksave(), !dbg !23
  store i8* %2, i8** %saved_stack, align 8, !dbg !23
  %vla = alloca i32, i64 %1, align 16, !dbg !23
  store i64 %1, i64* %__vla_expr0, align 8, !dbg !23
  call void @llvm.dbg.declare(metadata i64* %__vla_expr0, metadata !24, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i32* %vla, metadata !27, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata [10 x i32]* %array, metadata !32, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.declare(metadata %struct.S* %s, metadata !37, metadata !DIExpression()), !dbg !38
  call void @_ZN1SC1Ev(%struct.S* noundef nonnull align 4 dereferenceable(4) %s) #5, !dbg !38
  call void @_ZN1S3fooEv(%struct.S* noundef nonnull align 4 dereferenceable(4) %s), !dbg !39
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* %array, i64 0, i64 0, !dbg !40
  store i32 43, i32* %arrayidx, align 16, !dbg !41
  %arrayidx1 = getelementptr inbounds i32, i32* %vla, i64 0, !dbg !42
  store i32 43, i32* %arrayidx1, align 16, !dbg !43
  store i32 43, i32* %x, align 4, !dbg !44
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %x), "QUAL.OSS.FIRSTPRIVATE"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.FIRSTPRIVATE"([10 x i32]* %array), "QUAL.OSS.CAPTURED"(i64 %1) ], !dbg !45
  call void @llvm.dbg.declare(metadata i32* %x, metadata !20, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata i32* %vla, metadata !46, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata [10 x i32]* %array, metadata !32, metadata !DIExpression()), !dbg !36
  %4 = load i32, i32* %x, align 4, !dbg !50
  %inc = add nsw i32 %4, 1, !dbg !50
  store i32 %inc, i32* %x, align 4, !dbg !50
  %arrayidx2 = getelementptr inbounds i32, i32* %vla, i64 0, !dbg !52
  %5 = load i32, i32* %arrayidx2, align 16, !dbg !53
  %inc3 = add nsw i32 %5, 1, !dbg !53
  store i32 %inc3, i32* %arrayidx2, align 16, !dbg !53
  %arrayidx4 = getelementptr inbounds [10 x i32], [10 x i32]* %array, i64 0, i64 0, !dbg !54
  %6 = load i32, i32* %arrayidx4, align 16, !dbg !55
  %inc5 = add nsw i32 %6, 1, !dbg !55
  store i32 %inc5, i32* %arrayidx4, align 16, !dbg !55
  call void @llvm.directive.region.exit(token %3), !dbg !56
  %7 = load i8*, i8** %saved_stack, align 8, !dbg !57
  call void @llvm.stackrestore(i8* %7), !dbg !57
  ret i32 0, !dbg !57
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i8* @llvm.stacksave() #2

; Function Attrs: noinline nounwind optnone
declare void @_ZN1SC1Ev(%struct.S* noundef nonnull align 4 dereferenceable(4) %this)
; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr void @_ZN1S3fooEv(%struct.S* noundef nonnull align 4 dereferenceable(4) %this) #4 comdat align 2 !dbg !64 {
; CHECK:       final.then:
; CHECK-NEXT:    call void @llvm.dbg.declare(metadata %struct.S* %this
entry:
  %this.addr = alloca %struct.S*, align 8
  store %struct.S* %this, %struct.S** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.S** %this.addr, metadata !65, metadata !DIExpression()), !dbg !66
  %this1 = load %struct.S*, %struct.S** %this.addr, align 8
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %this1) ], !dbg !67
  call void @llvm.dbg.declare(metadata %struct.S* %this1, metadata !68, metadata !DIExpression()), !dbg !66
  %x = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0, !dbg !69
  %1 = load i32, i32* %x, align 4, !dbg !71
  %inc = add nsw i32 %1, 1, !dbg !71
  store i32 %inc, i32* %x, align 4, !dbg !71
  %x2 = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0, !dbg !72
  %2 = load i32, i32* %x2, align 4, !dbg !73
  %inc3 = add nsw i32 %2, 1, !dbg !73
  store i32 %inc3, i32* %x2, align 4, !dbg !73
  call void @llvm.directive.region.exit(token %0), !dbg !74
  ret void, !dbg !75
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #5

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore(i8*) #2

; Function Attrs: noinline nounwind optnone
declare void @_ZN1SC2Ev(%struct.S* noundef nonnull align 4 dereferenceable(4) %this)

; CHECH: define internal void @nanos6_unpacked_task_region_main0(i32* %x, i32* %vla, [10 x i32]* %array, i64 %0, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table)
; CHECK: call void @llvm.dbg.declare(metadata i32* %x
; CHECK-NEXT: call void @llvm.dbg.declare(metadata i32* %vla
; CHECK-NEXT: call void @llvm.dbg.declare(metadata [10 x i32]* %array

; CHECK: define internal void @nanos6_unpacked_task_region__ZN1S3fooEv0(%struct.S* %this1, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table)
; CHECK: call void @llvm.dbg.declare(metadata %struct.S* %this1

attributes #0 = { mustprogress noinline norecurse nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { nocallback nofree nosync nounwind willreturn }
attributes #3 = { noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #4 = { mustprogress noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "", checksumkind: CSK_MD5, checksum: "b87b255a8cd39b99cd4b6f68485b9f53")
!2 = !{!3}
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !4, line: 1, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !5, identifier: "_ZTS1S")
!4 = !DIFile(filename: "task_dbg.ll", directory: "", checksumkind: CSK_MD5, checksum: "b87b255a8cd39b99cd4b6f68485b9f53")
!5 = !{!6, !8}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !3, file: !4, line: 2, baseType: !7, size: 32)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DISubprogram(name: "foo", linkageName: "_ZN1S3fooEv", scope: !3, file: !4, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped, spFlags: 0)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 15.0.0 (git@bscpm03.bsc.es:llvm-ompss/llvm-mono.git c33e6ce010ca4044a427786e99566ffed9e8074b)"}
!16 = distinct !DISubprogram(name: "main", scope: !4, file: !4, line: 12, type: !17, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{!7}
!19 = !{}
!20 = !DILocalVariable(name: "x", scope: !16, file: !4, line: 13, type: !7)
!21 = !DILocation(line: 13, column: 9, scope: !16)
!22 = !DILocation(line: 14, column: 13, scope: !16)
!23 = !DILocation(line: 14, column: 5, scope: !16)
!24 = !DILocalVariable(name: "__vla_expr0", scope: !16, type: !25, flags: DIFlagArtificial)
!25 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!26 = !DILocation(line: 0, scope: !16)
!27 = !DILocalVariable(name: "vla", scope: !16, file: !4, line: 14, type: !28)
!28 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, elements: !29)
!29 = !{!30}
!30 = !DISubrange(count: !24)
!31 = !DILocation(line: 14, column: 9, scope: !16)
!32 = !DILocalVariable(name: "array", scope: !16, file: !4, line: 15, type: !33)
!33 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 320, elements: !34)
!34 = !{!35}
!35 = !DISubrange(count: 10)
!36 = !DILocation(line: 15, column: 9, scope: !16)
!37 = !DILocalVariable(name: "s", scope: !16, file: !4, line: 16, type: !3)
!38 = !DILocation(line: 16, column: 7, scope: !16)
!39 = !DILocation(line: 17, column: 7, scope: !16)
!40 = !DILocation(line: 18, column: 18, scope: !16)
!41 = !DILocation(line: 18, column: 27, scope: !16)
!42 = !DILocation(line: 18, column: 9, scope: !16)
!43 = !DILocation(line: 18, column: 16, scope: !16)
!44 = !DILocation(line: 18, column: 7, scope: !16)
!45 = !DILocation(line: 19, column: 13, scope: !16)
!46 = !DILocalVariable(name: "vla", scope: !16, file: !4, line: 14, type: !47)
!47 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, elements: !48)
!48 = !{!49}
!49 = !DISubrange(count: -1)
!50 = !DILocation(line: 21, column: 10, scope: !51)
!51 = distinct !DILexicalBlock(scope: !16, file: !4, line: 20, column: 5)
!52 = !DILocation(line: 22, column: 9, scope: !51)
!53 = !DILocation(line: 22, column: 15, scope: !51)
!54 = !DILocation(line: 23, column: 9, scope: !51)
!55 = !DILocation(line: 23, column: 17, scope: !51)
!56 = !DILocation(line: 24, column: 5, scope: !51)
!57 = !DILocation(line: 25, column: 1, scope: !16)
!58 = distinct !DISubprogram(name: "S", linkageName: "_ZN1SC1Ev", scope: !3, file: !4, line: 1, type: !9, scopeLine: 1, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !59, retainedNodes: !19)
!59 = !DISubprogram(name: "S", scope: !3, type: !9, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: 0)
!60 = !DILocalVariable(name: "this", arg: 1, scope: !58, type: !61, flags: DIFlagArtificial | DIFlagObjectPointer)
!61 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!62 = !DILocation(line: 0, scope: !58)
!63 = !DILocation(line: 1, column: 8, scope: !58)
!64 = distinct !DISubprogram(name: "foo", linkageName: "_ZN1S3fooEv", scope: !3, file: !4, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !8, retainedNodes: !19)
!65 = !DILocalVariable(name: "this", arg: 1, scope: !64, type: !61, flags: DIFlagArtificial | DIFlagObjectPointer)
!66 = !DILocation(line: 0, scope: !64)
!67 = !DILocation(line: 4, column: 17, scope: !64)
!68 = !DILocalVariable(name: "this", scope: !64, type: !3, flags: DIFlagArtificial)
!69 = !DILocation(line: 6, column: 13, scope: !70)
!70 = distinct !DILexicalBlock(scope: !64, file: !4, line: 5, column: 9)
!71 = !DILocation(line: 6, column: 14, scope: !70)
!72 = !DILocation(line: 7, column: 13, scope: !70)
!73 = !DILocation(line: 7, column: 14, scope: !70)
!74 = !DILocation(line: 8, column: 9, scope: !70)
!75 = !DILocation(line: 9, column: 5, scope: !64)
!76 = distinct !DISubprogram(name: "S", linkageName: "_ZN1SC2Ev", scope: !3, file: !4, line: 1, type: !9, scopeLine: 1, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !59, retainedNodes: !19)
!77 = !DILocalVariable(name: "this", arg: 1, scope: !76, type: !61, flags: DIFlagArtificial | DIFlagObjectPointer)
!78 = !DILocation(line: 0, scope: !76)
!79 = !DILocation(line: 2, column: 9, scope: !76)
!80 = !DILocation(line: 1, column: 8, scope: !76)
