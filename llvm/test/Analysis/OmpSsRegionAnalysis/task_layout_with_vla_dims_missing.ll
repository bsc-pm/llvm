; RUN: opt -ompss-2-regions -analyze -disable-checks -print-verbosity=dsa_vla_dims_missing < %s 2>&1 | FileCheck %s

define dso_local void @vla_senction_dep(i32 %n, i32 %k, i32 %j) {
entry:
  %n.addr = alloca i32, align 4
  %k.addr = alloca i32, align 4
  %j.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  %__vla_expr1 = alloca i64, align 8
  %__vla_expr2 = alloca i64, align 8
  store i32 %n, i32* %n.addr, align 4
  store i32 %k, i32* %k.addr, align 4
  store i32 %j, i32* %j.addr, align 4
  %0 = load i32, i32* %n.addr, align 4
  %add = add nsw i32 %0, 1
  %1 = zext i32 %add to i64
  %2 = load i32, i32* %k.addr, align 4
  %add1 = add nsw i32 %2, 2
  %3 = zext i32 %add1 to i64
  %4 = load i32, i32* %j.addr, align 4
  %add2 = add nsw i32 %4, 3
  %5 = zext i32 %add2 to i64
  %6 = call i8* @llvm.stacksave()
  store i8* %6, i8** %saved_stack, align 8
  %7 = mul nuw i64 %1, %3
  %8 = mul nuw i64 %7, %5
  %vla = alloca i32, i64 %8, align 16
  store i64 %1, i64* %__vla_expr0, align 8
  store i64 %3, i64* %__vla_expr1, align 8
  store i64 %5, i64* %__vla_expr2, align 8
  %9 = mul i64 %5, 4
  %10 = mul i64 %5, 4
  %11 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.DEP.IN"(i32* %vla, i64 %9, i64 0, i64 %10, i64 %3, i64 0, i64 %3, i64 %1, i64 0, i64 %1) ]
  call void @llvm.directive.region.exit(token %11)
  %12 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.DEP.IN"(i32* %vla, i64 %9, i64 0, i64 %10, i64 %3, i64 0, i64 %3, i64 %1, i64 0, i64 %1) ]
  call void @llvm.directive.region.exit(token %12)
  %13 = load i8*, i8** %saved_stack, align 8
  call void @llvm.stackrestore(i8* %13)
  ret void
}

; CHECK: [0] %11
; CHECK-NEXT:   %vla
; CHECK-NEXT: [0] %12
; CHECK-NEXT:   %vla

declare i8* @llvm.stacksave()
declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)
declare void @llvm.stackrestore(i8*)
