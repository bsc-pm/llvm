; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_reduction.c'
source_filename = "task_reduction.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; #pragma oss declare reduction(asdf: int : omp_out += omp_in) initializer(omp_priv = 0)
;
; void foo(int n) {
;     int array[n];
;     #pragma oss task reduction(asdf : n, array)
;     {}
; }
;
; void foo1(int n) {
;     int array[n];
;     #pragma oss task reduction(+ : n, array)
;     {}
; }
;
; int main() {
;     foo(3);
;     foo1(3);
; }

%struct._depend_unpack_t = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.0 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.1 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.2 = type { i32*, i64, i64, i64 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo(i32 %n) #0 !dbg !6 {
entry:
  %n.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32, i32* %n.addr, align 4, !dbg !8
  %1 = zext i32 %0 to i64, !dbg !9
  %2 = call i8* @llvm.stacksave(), !dbg !9
  store i8* %2, i8** %saved_stack, align 8, !dbg !9
  %vla = alloca i32, i64 %1, align 16, !dbg !9
  store i64 %1, i64* %__vla_expr0, align 8, !dbg !9
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %n.addr), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1), "QUAL.OSS.DEP.REDUCTION"(i32 -1, i32* %n.addr, %struct._depend_unpack_t (i32*)* @compute_dep, i32* %n.addr), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %n.addr, void (i32*, i32*, i64)* @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %n.addr, void (i32*, i32*, i64)* @red_comb), "QUAL.OSS.DEP.REDUCTION"(i32 -1, i32* %vla, %struct._depend_unpack_t.0 (i32*, i64)* @compute_dep.1, i32* %vla, i64 %1), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %vla, void (i32*, i32*, i64)* @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %vla, void (i32*, i32*, i64)* @red_comb) ], !dbg !10
  call void @llvm.directive.region.exit(token %3), !dbg !11
  %4 = load i8*, i8** %saved_stack, align 8, !dbg !12
  call void @llvm.stackrestore(i8* %4), !dbg !12
  ret void, !dbg !12
}

; CHECK: define internal void @nanos6_ol_task_region_foo0(%nanos6_task_args_foo0* %task_args, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_n.addr = getelementptr %nanos6_task_args_foo0, %nanos6_task_args_foo0* %task_args, i32 0, i32 0
; CHECK-NEXT:   %load_gep_n.addr = load i32*, i32** %gep_n.addr
; CHECK-NEXT:   %gep_vla = getelementptr %nanos6_task_args_foo0, %nanos6_task_args_foo0* %task_args, i32 0, i32 1
; CHECK-NEXT:   %load_gep_vla = load i32*, i32** %gep_vla
; CHECK-NEXT:   %capt_gep = getelementptr %nanos6_task_args_foo0, %nanos6_task_args_foo0* %task_args, i32 0, i32 2
; CHECK-NEXT:   %load_capt_gep = load i64, i64* %capt_gep
; CHECK-NEXT:   %0 = call %struct._depend_unpack_t @compute_dep(i32* %load_gep_n.addr)
; CHECK-NEXT:   %1 = extractvalue %struct._depend_unpack_t %0, 0
; CHECK-NEXT:   %2 = alloca i32*
; CHECK-NEXT:   %local_lookup_n.addr = getelementptr %nanos6_address_translation_entry_t, %nanos6_address_translation_entry_t* %address_translation_table, i32 0, i32 0
; CHECK-NEXT:   %3 = load i64, i64* %local_lookup_n.addr
; CHECK-NEXT:   %device_lookup_n.addr = getelementptr %nanos6_address_translation_entry_t, %nanos6_address_translation_entry_t* %address_translation_table, i32 0, i32 1
; CHECK-NEXT:   %4 = load i64, i64* %device_lookup_n.addr
; CHECK-NEXT:   %5 = bitcast i32* %1 to i8*
; CHECK-NEXT:   %6 = sub i64 0, %3
; CHECK-NEXT:   %7 = getelementptr i8, i8* %5, i64 %6
; CHECK-NEXT:   %8 = getelementptr i8, i8* %7, i64 %4
; CHECK-NEXT:   %9 = bitcast i8* %8 to i32*
; CHECK-NEXT:   store i32* %9, i32** %2
; CHECK-NEXT:   %10 = load i32*, i32** %2
; CHECK-NEXT:   %11 = call %struct._depend_unpack_t.0 @compute_dep.1(i32* %load_gep_vla, i64 %load_capt_gep)
; CHECK-NEXT:   %12 = extractvalue %struct._depend_unpack_t.0 %11, 0
; CHECK-NEXT:   %13 = alloca i32*
; CHECK-NEXT:   %local_lookup_vla = getelementptr %nanos6_address_translation_entry_t, %nanos6_address_translation_entry_t* %address_translation_table, i32 1, i32 0
; CHECK-NEXT:   %14 = load i64, i64* %local_lookup_vla
; CHECK-NEXT:   %device_lookup_vla = getelementptr %nanos6_address_translation_entry_t, %nanos6_address_translation_entry_t* %address_translation_table, i32 1, i32 1
; CHECK-NEXT:   %15 = load i64, i64* %device_lookup_vla
; CHECK-NEXT:   %16 = bitcast i32* %12 to i8*
; CHECK-NEXT:   %17 = sub i64 0, %14
; CHECK-NEXT:   %18 = getelementptr i8, i8* %16, i64 %17
; CHECK-NEXT:   %19 = getelementptr i8, i8* %18, i64 %15
; CHECK-NEXT:   %20 = bitcast i8* %19 to i32*
; CHECK-NEXT:   store i32* %20, i32** %13
; CHECK-NEXT:   %21 = load i32*, i32** %13
; CHECK-NEXT:   call void @nanos6_unpacked_task_region_foo0(i32* %10, i32* %21, i64 %load_capt_gep, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_deps_foo0(i32* %n.addr, i32* %vla, i64 %0, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = call %struct._depend_unpack_t @compute_dep(i32* %n.addr)
; CHECK-NEXT:   %2 = call %struct._depend_unpack_t @compute_dep(i32* %n.addr)
; CHECK-NEXT:   %3 = extractvalue %struct._depend_unpack_t %1, 0
; CHECK-NEXT:   %4 = bitcast i32* %3 to i8*
; CHECK-NEXT:   %5 = extractvalue %struct._depend_unpack_t %1, 1
; CHECK-NEXT:   %6 = extractvalue %struct._depend_unpack_t %1, 2
; CHECK-NEXT:   %7 = extractvalue %struct._depend_unpack_t %2, 3
; CHECK-NEXT:   call void @nanos6_register_region_reduction_depinfo1(i32 -1, i32 0, i8* %handler, i32 0, i8* null, i8* %4, i64 %5, i64 %6, i64 %7)
; CHECK-NEXT:   %8 = call %struct._depend_unpack_t.0 @compute_dep.1(i32* %vla, i64 %0)
; CHECK-NEXT:   %9 = call %struct._depend_unpack_t.0 @compute_dep.1(i32* %vla, i64 %0)
; CHECK-NEXT:   %10 = extractvalue %struct._depend_unpack_t.0 %8, 0
; CHECK-NEXT:   %11 = bitcast i32* %10 to i8*
; CHECK-NEXT:   %12 = extractvalue %struct._depend_unpack_t.0 %8, 1
; CHECK-NEXT:   %13 = extractvalue %struct._depend_unpack_t.0 %8, 2
; CHECK-NEXT:   %14 = extractvalue %struct._depend_unpack_t.0 %9, 3
; CHECK-NEXT:   call void @nanos6_register_region_reduction_depinfo1(i32 -1, i32 0, i8* %handler, i32 1, i8* null, i8* %11, i64 %12, i64 %13, i64 %14)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #1

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

; Function Attrs: noinline norecurse nounwind uwtable
define internal void @red_init(i32* %0, i32* %1, i64 %2) #2 !dbg !13 {
entry:
  %.addr = alloca i32*, align 8
  %.addr1 = alloca i32*, align 8
  %.addr2 = alloca i64, align 8
  store i32* %0, i32** %.addr, align 8
  store i32* %1, i32** %.addr1, align 8
  store i64 %2, i64* %.addr2, align 8
  %3 = load i32*, i32** %.addr, align 8
  %4 = load i32*, i32** %.addr1, align 8
  %5 = load i64, i64* %.addr2, align 8
  %6 = udiv exact i64 %5, 4
  %arrayctor.dst.end = getelementptr inbounds i32, i32* %3, i64 %6
  br label %arrayctor.loop

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi i32* [ %3, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ]
  %arrayctor.src.cur = phi i32* [ %4, %entry ], [ %arrayctor.src.next, %arrayctor.loop ]
  store i32 0, i32* %3, align 4
  %arrayctor.dst.next = getelementptr inbounds i32, i32* %arrayctor.dst.cur, i64 1
  %arrayctor.src.next = getelementptr inbounds i32, i32* %arrayctor.src.cur, i64 1
  %arrayctor.done = icmp eq i32* %arrayctor.dst.next, %arrayctor.dst.end
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !14
}

; Function Attrs: noinline norecurse nounwind uwtable
define internal void @red_comb(i32* %0, i32* %1, i64 %2) #2 !dbg !15 {
entry:
  %.addr = alloca i32*, align 8
  %.addr1 = alloca i32*, align 8
  %.addr2 = alloca i64, align 8
  store i32* %0, i32** %.addr, align 8
  store i32* %1, i32** %.addr1, align 8
  store i64 %2, i64* %.addr2, align 8
  %3 = load i32*, i32** %.addr, align 8
  %4 = load i32*, i32** %.addr1, align 8
  %5 = load i64, i64* %.addr2, align 8
  %6 = udiv exact i64 %5, 4
  %arrayctor.dst.end = getelementptr inbounds i32, i32* %3, i64 %6
  br label %arrayctor.loop

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi i32* [ %3, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ]
  %arrayctor.src.cur = phi i32* [ %4, %entry ], [ %arrayctor.src.next, %arrayctor.loop ]
  %7 = load i32, i32* %arrayctor.src.cur, align 4, !dbg !16
  %8 = load i32, i32* %arrayctor.dst.cur, align 4, !dbg !17
  %add = add nsw i32 %8, %7, !dbg !17
  store i32 %add, i32* %arrayctor.dst.cur, align 4, !dbg !17
  %arrayctor.dst.next = getelementptr inbounds i32, i32* %arrayctor.dst.cur, i64 1
  %arrayctor.src.next = getelementptr inbounds i32, i32* %arrayctor.src.cur, i64 1
  %arrayctor.done = icmp eq i32* %arrayctor.dst.next, %arrayctor.dst.end
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !18
}

define internal %struct._depend_unpack_t @compute_dep(i32* %n.addr) {
entry:
  %return.val = alloca %struct._depend_unpack_t, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 0
  store i32* %n.addr, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 1
  store i64 4, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 3
  store i64 4, i64* %3, align 8
  %4 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, align 8
  ret %struct._depend_unpack_t %4
}

define internal %struct._depend_unpack_t.0 @compute_dep.1(i32* %vla, i64 %0) {
entry:
  %return.val = alloca %struct._depend_unpack_t.0, align 8
  %1 = mul i64 %0, 4
  %2 = mul i64 %0, 4
  %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 0
  store i32* %vla, i32** %3, align 8
  %4 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 1
  store i64 %1, i64* %4, align 8
  %5 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 2
  store i64 0, i64* %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 3
  store i64 %2, i64* %6, align 8
  %7 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, align 8
  ret %struct._depend_unpack_t.0 %7
}

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo1(i32 %n) #0 !dbg !19 {
entry:
  %n.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32, i32* %n.addr, align 4, !dbg !20
  %1 = zext i32 %0 to i64, !dbg !21
  %2 = call i8* @llvm.stacksave(), !dbg !21
  store i8* %2, i8** %saved_stack, align 8, !dbg !21
  %vla = alloca i32, i64 %1, align 16, !dbg !21
  store i64 %1, i64* %__vla_expr0, align 8, !dbg !21
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %n.addr), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1), "QUAL.OSS.DEP.REDUCTION"(i32 6000, i32* %n.addr, %struct._depend_unpack_t.1 (i32*)* @compute_dep.4, i32* %n.addr), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %n.addr, void (i32*, i32*, i64)* @red_init.2), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %n.addr, void (i32*, i32*, i64)* @red_comb.3), "QUAL.OSS.DEP.REDUCTION"(i32 6000, i32* %vla, %struct._depend_unpack_t.2 (i32*, i64)* @compute_dep.5, i32* %vla, i64 %1), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %vla, void (i32*, i32*, i64)* @red_init.2), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %vla, void (i32*, i32*, i64)* @red_comb.3) ], !dbg !22
  call void @llvm.directive.region.exit(token %3), !dbg !23
  %4 = load i8*, i8** %saved_stack, align 8, !dbg !24
  call void @llvm.stackrestore(i8* %4), !dbg !24
  ret void, !dbg !24
}

; CHECK: define internal void @nanos6_ol_task_region_foo10(%nanos6_task_args_foo10* %task_args, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_n.addr = getelementptr %nanos6_task_args_foo10, %nanos6_task_args_foo10* %task_args, i32 0, i32 0
; CHECK-NEXT:   %load_gep_n.addr = load i32*, i32** %gep_n.addr
; CHECK-NEXT:   %gep_vla = getelementptr %nanos6_task_args_foo10, %nanos6_task_args_foo10* %task_args, i32 0, i32 1
; CHECK-NEXT:   %load_gep_vla = load i32*, i32** %gep_vla
; CHECK-NEXT:   %capt_gep = getelementptr %nanos6_task_args_foo10, %nanos6_task_args_foo10* %task_args, i32 0, i32 2
; CHECK-NEXT:   %load_capt_gep = load i64, i64* %capt_gep
; CHECK-NEXT:   %0 = call %struct._depend_unpack_t.1 @compute_dep.4(i32* %load_gep_n.addr)
; CHECK-NEXT:   %1 = extractvalue %struct._depend_unpack_t.1 %0, 0
; CHECK-NEXT:   %2 = alloca i32*
; CHECK-NEXT:   %local_lookup_n.addr = getelementptr %nanos6_address_translation_entry_t, %nanos6_address_translation_entry_t* %address_translation_table, i32 0, i32 0
; CHECK-NEXT:   %3 = load i64, i64* %local_lookup_n.addr
; CHECK-NEXT:   %device_lookup_n.addr = getelementptr %nanos6_address_translation_entry_t, %nanos6_address_translation_entry_t* %address_translation_table, i32 0, i32 1
; CHECK-NEXT:   %4 = load i64, i64* %device_lookup_n.addr
; CHECK-NEXT:   %5 = bitcast i32* %1 to i8*
; CHECK-NEXT:   %6 = sub i64 0, %3
; CHECK-NEXT:   %7 = getelementptr i8, i8* %5, i64 %6
; CHECK-NEXT:   %8 = getelementptr i8, i8* %7, i64 %4
; CHECK-NEXT:   %9 = bitcast i8* %8 to i32*
; CHECK-NEXT:   store i32* %9, i32** %2
; CHECK-NEXT:   %10 = load i32*, i32** %2
; CHECK-NEXT:   %11 = call %struct._depend_unpack_t.2 @compute_dep.5(i32* %load_gep_vla, i64 %load_capt_gep)
; CHECK-NEXT:   %12 = extractvalue %struct._depend_unpack_t.2 %11, 0
; CHECK-NEXT:   %13 = alloca i32*
; CHECK-NEXT:   %local_lookup_vla = getelementptr %nanos6_address_translation_entry_t, %nanos6_address_translation_entry_t* %address_translation_table, i32 1, i32 0
; CHECK-NEXT:   %14 = load i64, i64* %local_lookup_vla
; CHECK-NEXT:   %device_lookup_vla = getelementptr %nanos6_address_translation_entry_t, %nanos6_address_translation_entry_t* %address_translation_table, i32 1, i32 1
; CHECK-NEXT:   %15 = load i64, i64* %device_lookup_vla
; CHECK-NEXT:   %16 = bitcast i32* %12 to i8*
; CHECK-NEXT:   %17 = sub i64 0, %14
; CHECK-NEXT:   %18 = getelementptr i8, i8* %16, i64 %17
; CHECK-NEXT:   %19 = getelementptr i8, i8* %18, i64 %15
; CHECK-NEXT:   %20 = bitcast i8* %19 to i32*
; CHECK-NEXT:   store i32* %20, i32** %13
; CHECK-NEXT:   %21 = load i32*, i32** %13
; CHECK-NEXT:   call void @nanos6_unpacked_task_region_foo10(i32* %10, i32* %21, i64 %load_capt_gep, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_deps_foo10(i32* %n.addr, i32* %vla, i64 %0, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = call %struct._depend_unpack_t.1 @compute_dep.4(i32* %n.addr)
; CHECK-NEXT:   %2 = call %struct._depend_unpack_t.1 @compute_dep.4(i32* %n.addr)
; CHECK-NEXT:   %3 = extractvalue %struct._depend_unpack_t.1 %1, 0
; CHECK-NEXT:   %4 = bitcast i32* %3 to i8*
; CHECK-NEXT:   %5 = extractvalue %struct._depend_unpack_t.1 %1, 1
; CHECK-NEXT:   %6 = extractvalue %struct._depend_unpack_t.1 %1, 2
; CHECK-NEXT:   %7 = extractvalue %struct._depend_unpack_t.1 %2, 3
; CHECK-NEXT:   call void @nanos6_register_region_reduction_depinfo1(i32 6000, i32 0, i8* %handler, i32 0, i8* null, i8* %4, i64 %5, i64 %6, i64 %7)
; CHECK-NEXT:   %8 = call %struct._depend_unpack_t.2 @compute_dep.5(i32* %vla, i64 %0)
; CHECK-NEXT:   %9 = call %struct._depend_unpack_t.2 @compute_dep.5(i32* %vla, i64 %0)
; CHECK-NEXT:   %10 = extractvalue %struct._depend_unpack_t.2 %8, 0
; CHECK-NEXT:   %11 = bitcast i32* %10 to i8*
; CHECK-NEXT:   %12 = extractvalue %struct._depend_unpack_t.2 %8, 1
; CHECK-NEXT:   %13 = extractvalue %struct._depend_unpack_t.2 %8, 2
; CHECK-NEXT:   %14 = extractvalue %struct._depend_unpack_t.2 %9, 3
; CHECK-NEXT:   call void @nanos6_register_region_reduction_depinfo1(i32 6000, i32 0, i8* %handler, i32 1, i8* null, i8* %11, i64 %12, i64 %13, i64 %14)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; Function Attrs: noinline norecurse nounwind uwtable
define internal void @red_init.2(i32* %0, i32* %1, i64 %2) #2 !dbg !25 {
entry:
  %.addr = alloca i32*, align 8
  %.addr1 = alloca i32*, align 8
  %.addr2 = alloca i64, align 8
  store i32* %0, i32** %.addr, align 8
  store i32* %1, i32** %.addr1, align 8
  store i64 %2, i64* %.addr2, align 8
  %3 = load i32*, i32** %.addr, align 8
  %4 = load i32*, i32** %.addr1, align 8
  %5 = load i64, i64* %.addr2, align 8
  %6 = udiv exact i64 %5, 4
  %arrayctor.dst.end = getelementptr inbounds i32, i32* %3, i64 %6
  br label %arrayctor.loop

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi i32* [ %3, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ]
  %arrayctor.src.cur = phi i32* [ %4, %entry ], [ %arrayctor.src.next, %arrayctor.loop ]
  store i32 0, i32* %3, align 4
  %arrayctor.dst.next = getelementptr inbounds i32, i32* %arrayctor.dst.cur, i64 1
  %arrayctor.src.next = getelementptr inbounds i32, i32* %arrayctor.src.cur, i64 1
  %arrayctor.done = icmp eq i32* %arrayctor.dst.next, %arrayctor.dst.end
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !26
}

; Function Attrs: noinline norecurse nounwind uwtable
define internal void @red_comb.3(i32* %0, i32* %1, i64 %2) #2 !dbg !27 {
entry:
  %.addr = alloca i32*, align 8
  %.addr1 = alloca i32*, align 8
  %.addr2 = alloca i64, align 8
  store i32* %0, i32** %.addr, align 8
  store i32* %1, i32** %.addr1, align 8
  store i64 %2, i64* %.addr2, align 8
  %3 = load i32*, i32** %.addr, align 8
  %4 = load i32*, i32** %.addr1, align 8
  %5 = load i64, i64* %.addr2, align 8
  %6 = udiv exact i64 %5, 4
  %arrayctor.dst.end = getelementptr inbounds i32, i32* %3, i64 %6
  br label %arrayctor.loop

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi i32* [ %3, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ]
  %arrayctor.src.cur = phi i32* [ %4, %entry ], [ %arrayctor.src.next, %arrayctor.loop ]
  %7 = load i32, i32* %arrayctor.dst.cur, align 4, !dbg !28
  %8 = load i32, i32* %arrayctor.src.cur, align 4, !dbg !28
  %add = add nsw i32 %7, %8, !dbg !29
  store i32 %add, i32* %arrayctor.dst.cur, align 4, !dbg !29
  %arrayctor.dst.next = getelementptr inbounds i32, i32* %arrayctor.dst.cur, i64 1
  %arrayctor.src.next = getelementptr inbounds i32, i32* %arrayctor.src.cur, i64 1
  %arrayctor.done = icmp eq i32* %arrayctor.dst.next, %arrayctor.dst.end
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !28
}

define internal %struct._depend_unpack_t.1 @compute_dep.4(i32* %n.addr) {
entry:
  %return.val = alloca %struct._depend_unpack_t.1, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 0
  store i32* %n.addr, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 1
  store i64 4, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 3
  store i64 4, i64* %3, align 8
  %4 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, align 8
  ret %struct._depend_unpack_t.1 %4
}

define internal %struct._depend_unpack_t.2 @compute_dep.5(i32* %vla, i64 %0) {
entry:
  %return.val = alloca %struct._depend_unpack_t.2, align 8
  %1 = mul i64 %0, 4
  %2 = mul i64 %0, 4
  %3 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %return.val, i32 0, i32 0
  store i32* %vla, i32** %3, align 8
  %4 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %return.val, i32 0, i32 1
  store i64 %1, i64* %4, align 8
  %5 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %return.val, i32 0, i32 2
  store i64 0, i64* %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %return.val, i32 0, i32 3
  store i64 %2, i64* %6, align 8
  %7 = load %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %return.val, align 8
  ret %struct._depend_unpack_t.2 %7
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !30 {
entry:
  call void @foo(i32 3), !dbg !31
  call void @foo1(i32 3), !dbg !32
  ret i32 0, !dbg !33
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "task_reduction.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !7, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 4, column: 15, scope: !6)
!9 = !DILocation(line: 4, column: 5, scope: !6)
!10 = !DILocation(line: 5, column: 13, scope: !6)
!11 = !DILocation(line: 6, column: 6, scope: !6)
!12 = !DILocation(line: 7, column: 1, scope: !6)
!13 = distinct !DISubprogram(linkageName: "red_init", scope: !1, file: !1, type: !7, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!14 = !DILocation(line: 1, column: 85, scope: !13)
!15 = distinct !DISubprogram(linkageName: "red_comb", scope: !1, file: !1, type: !7, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 1, column: 54, scope: !15)
!17 = !DILocation(line: 1, column: 51, scope: !15)
!18 = !DILocation(line: 1, column: 43, scope: !15)
!19 = distinct !DISubprogram(name: "foo1", scope: !1, file: !1, line: 9, type: !7, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!20 = !DILocation(line: 10, column: 15, scope: !19)
!21 = !DILocation(line: 10, column: 5, scope: !19)
!22 = !DILocation(line: 11, column: 13, scope: !19)
!23 = !DILocation(line: 12, column: 6, scope: !19)
!24 = !DILocation(line: 13, column: 1, scope: !19)
!25 = distinct !DISubprogram(linkageName: "red_init.1", scope: !1, file: !1, type: !7, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!26 = !DILocation(line: 11, column: 36, scope: !25)
!27 = distinct !DISubprogram(linkageName: "red_comb.2", scope: !1, file: !1, type: !7, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!28 = !DILocation(line: 11, column: 36, scope: !27)
!29 = !DILocation(line: 11, column: 32, scope: !27)
!30 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 15, type: !7, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!31 = !DILocation(line: 16, column: 5, scope: !30)
!32 = !DILocation(line: 17, column: 5, scope: !30)
!33 = !DILocation(line: 18, column: 1, scope: !30)
