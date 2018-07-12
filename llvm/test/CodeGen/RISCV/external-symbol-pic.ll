; RUN: llc < %s | FileCheck %s --check-prefix=CHECK-NOPIC
; RUN: llc -relocation-model=pic < %s | FileCheck %s --check-prefix=CHECK-PIC

define signext i32 @foo(i32 signext %a, i32 signext %b) {
entry:
  ; CHECK-NOPIC: call  __divdi3
  ; CHECK-PIC: call  __divdi3@plt

  %div = sdiv i32 %a, %b
  ret i32 %div
}

