; RUN: opt -ompss-2-regions -analyze -disable-checks -print-verbosity=unpack < %s 2>&1 | FileCheck %s -check-prefix=UNPACK

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.S = type { [10 x i32] }

@global = common dso_local global i32 0, align 4

define dso_local void @foo(i32 %i, i32 %j) {
entry:
  %i.addr = alloca i32, align 4
  %j.addr = alloca i32, align 4
  %s = alloca %struct.S, align 4
  store i32 %i, i32* %i.addr, align 4
  store i32 %j, i32* %j.addr, align 4
; store from here
  %0 = load i32, i32* %i.addr, align 4
  %1 = load i32, i32* %j.addr, align 4
  %add = add nsw i32 %0, %1
  %2 = sext i32 %add to i64
  %3 = add i64 %2, 1
  %a = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 0
  %arraydecay = getelementptr inbounds [10 x i32], [10 x i32]* %a, i64 0, i64 0
  %4 = mul i64 %2, 4
  %5 = mul i64 %3, 4
; to here
  %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %s), "QUAL.OSS.SHARED"(i32* @global), "QUAL.OSS.FIRSTPRIVATE"(i32* %i.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %j.addr), "QUAL.OSS.DEP.IN"(i32* %arraydecay, i64 4, i64 %4, i64 %5), "QUAL.OSS.DEP.OUT"(i32* @global, i64 4, i64 0, i64 4) ]
    %a1 = alloca i32, align 4
    %7 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %a1), "QUAL.OSS.DEP.INOUT"(i32* %a1, i64 4, i64 0, i64 4) ]
    call void @llvm.directive.region.exit(token %7)
  call void @llvm.directive.region.exit(token %6)
  ret void
}

; UNPACK: [0] %6
; UNPACK-NEXT:   [Inst] %0
; UNPACK-NEXT:   [Inst] %1
; UNPACK-NEXT:   [Inst] %add
; UNPACK-NEXT:   [Inst] %2
; UNPACK-NEXT:   [Inst] %3
; UNPACK-NEXT:   [Inst] %a
; UNPACK-NEXT:   [Inst] %arraydecay
; UNPACK-NEXT:   [Inst] %4
; UNPACK-NEXT:   [Inst] %5
; UNPACK-NEXT:   [1] %7

@mat = common dso_local global [10 x [10 x i32]] zeroinitializer, align 16
@vec = common dso_local global [5 x i32] zeroinitializer, align 16

define dso_local void @foo1(i32 %i) {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @vec, i64 1, i64 0), align 4
  %1 = sext i32 %0 to i64
  %2 = add i64 %1, 1
  %3 = load i32, i32* %i.addr, align 4
  %add = add nsw i32 %3, %3
  %4 = sext i32 %add to i64
  %5 = add i64 %4, 1
  %6 = mul i64 %1, 4
  %7 = mul i64 %2, 4
  %8 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [10 x i32]]* @mat), "QUAL.OSS.FIRSTPRIVATE"(i32* %i.addr), "QUAL.OSS.FIRSTPRIVATE"([5 x i32]* @vec), "QUAL.OSS.DEP.IN"([10 x i32]* getelementptr inbounds ([10 x [10 x i32]], [10 x [10 x i32]]* @mat, i64 0, i64 0), i64 40, i64 %6, i64 %7, i64 10, i64 %4, i64 %5) ]
  call void @llvm.directive.region.exit(token %8)
  ret void
}

; UNPACK: [0] %8
; UNPACK-NEXT:   [Inst] %0
; UNPACK-NEXT:   [Inst] %1
; UNPACK-NEXT:   [Inst] %2
; UNPACK-NEXT:   [Inst] %3
; UNPACK-NEXT:   [Inst] %add
; UNPACK-NEXT:   [Inst] %4
; UNPACK-NEXT:   [Inst] %5
; UNPACK-NEXT:   [Inst] %6
; UNPACK-NEXT:   [Inst] %7
; UNPACK-NEXT:   [Const] getelementptr inbounds ([10 x [10 x i32]], [10 x [10 x i32]]* @mat, i64 0, i64 0)
; UNPACK-NEXT:   [Const] getelementptr inbounds ([5 x i32], [5 x i32]* @vec, i64 1, i64 0)

define dso_local void @vla_senction_dep(i32 %n, i32 %k, i32 %j) {
entry:
  %n.addr = alloca i32, align 4
  %k.addr = alloca i32, align 4
  %j.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  store i32 %n, i32* %n.addr, align 4
  store i32 %k, i32* %k.addr, align 4
  store i32 %j, i32* %j.addr, align 4
  %0 = load i32, i32* %n.addr, align 4
  %add = add nsw i32 %0, 1
  %1 = zext i32 %add to i64
  %2 = call i8* @llvm.stacksave()
  store i8* %2, i8** %saved_stack, align 8
  %vla = alloca i32, i64 %1, align 16
  store i64 %1, i64* %__vla_expr0, align 8
; unpack from here
  %3 = load i32, i32* %k.addr, align 4
  %4 = sext i32 %3 to i64
  %5 = load i32, i32* %j.addr, align 4
  %6 = sext i32 %5 to i64
  %7 = add i64 1, %6
  %8 = mul i64 %1, 4
  %9 = mul i64 %4, 4
  %10 = mul i64 %7, 4
; to here
  %11 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1), "QUAL.OSS.FIRSTPRIVATE"(i32* %k.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %j.addr), "QUAL.OSS.DEP.IN"(i32* %vla, i64 %8, i64 %9, i64 %10) ]
  call void @llvm.directive.region.exit(token %11)
  %12 = load i8*, i8** %saved_stack, align 8
  call void @llvm.stackrestore(i8* %12)
  ret void
}

; UNPACK: [0] %11
; UNPACK-NEXT:   [Inst] %3
; UNPACK-NEXT:   [Inst] %4
; UNPACK-NEXT:   [Inst] %5
; UNPACK-NEXT:   [Inst] %6
; UNPACK-NEXT:   [Inst] %7
; UNPACK-NEXT:   [Inst] %8
; UNPACK-NEXT:   [Inst] %9
; UNPACK-NEXT:   [Inst] %10

declare i8* @llvm.stacksave()
declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)
declare void @llvm.stackrestore(i8*)

