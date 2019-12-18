; RUN: opt -ompss-2-regions -analyze -disable-checks -print-verbosity=uses < %s 2>&1 | FileCheck %s -check-prefix=USES
; RUN: opt -ompss-2-regions -analyze -disable-checks -print-verbosity=dsa_missing < %s 2>&1 | FileCheck %s -check-prefix=DSA

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @foo1(i32 %arg) {
entry:
  %func1 = add nsw i32 %arg, 1
  %func2 = alloca i32, align 4
  %func3 = alloca i32, align 4
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
    %task0_1 = add nsw i32 %func1, 1
    %task0_2 = add nsw i32 %task0_1, 1
    %task0_3 = add nsw i32 %arg, %task0_2

    %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %func2) ]
      %task1_1 = load i32, i32* %func2, align 4
      %task1_2 = load i32, i32* %func3, align 4
    call void @llvm.directive.region.exit(token %1)

    %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
      %task2_1 = add nsw i32 %task0_1, 1
    call void @llvm.directive.region.exit(token %2)

    %task0_4 = add nsw i32 %task2_1, 1

  call void @llvm.directive.region.exit(token %0)

  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
    %task3_1 = add nsw i32 %task0_4, 1
  call void @llvm.directive.region.exit(token %3)

  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
  call void @llvm.directive.region.exit(token %4)
  ret i32 0

; USES: [0] %0
; USES-NEXT:   [Before] %func1
; USES-NEXT:   [Before] %arg
; USES-NEXT:   [After] %task0_4
; USES-NEXT:   [1] %1
; USES-NEXT:     [Before] %func2
; USES-NEXT:     [Before] %func3
; USES-NEXT:   [1] %2
; USES-NEXT:     [Before] %task0_1
; USES-NEXT:     [After] %task2_1
; USES-NEXT: [0] %3
; USES-NEXT:   [Before] %task0_4
; USES-NEXT: [0] %4

; DSA: [0] %0
; DSA-NEXT:   %func1
; DSA-NEXT:   %arg
; DSA-NEXT:   [1] %1
; DSA-NEXT:     %func3
; DSA-NEXT:   [1] %2
; DSA-NEXT:     %task0_1
; DSA-NEXT: [0] %3
; DSA-NEXT:   %task0_4
; DSA-NEXT: [0] %4

}

define i32 @foo2(i32 %arg) {
entry:
  %func1 = add nsw i32 %arg, 1
  %func2 = alloca i32, align 4
  %func3 = alloca i32, align 4
  br i1 1, label %if.then, label %if.else

if.then:
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %func2) ]
    %task0_1 = add nsw i32 %func1, 1
    %task0_2 = add nsw i32 %task0_1, 1
    %task0_3 = add nsw i32 %arg, %task0_2
  call void @llvm.directive.region.exit(token %0)
  br label %if.end

if.else:
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %func2) ]
    %task1_1 = load i32, i32* %func2, align 4
    %task1_2 = load i32, i32* %func3, align 4
    %task1_3 = add nsw i32 %arg, %task1_2
  call void @llvm.directive.region.exit(token %1)
  br label %if.end

if.end:
  ret i32 0

; USES: [0] %0
; USES-NEXT:   [Before] %func1
; USES-NEXT:   [Before] %arg
; USES-NEXT: [0] %1
; USES-NEXT:   [Before] %func2
; USES-NEXT:   [Before] %func3
; USES-NEXT:   [Before] %arg

; DSA: [0] %0
; DSA-NEXT:   %func1
; DSA-NEXT:   %arg
; DSA-NEXT: [0] %1
; DSA-NEXT:   %func3
; DSA-NEXT:   %arg

}

declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)

