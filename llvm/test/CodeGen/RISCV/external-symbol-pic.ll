; RUN: llc -mtriple=riscv32 < %s | FileCheck %s --check-prefix=CHECK-NOPIC-RV32
; RUN: llc -mtriple=riscv32 -relocation-model=pic < %s | FileCheck %s --check-prefix=CHECK-PIC-RV32
; RUN: llc -mtriple=riscv64 < %s | FileCheck %s --check-prefix=CHECK-NOPIC-RV64
; RUN: llc -mtriple=riscv64 -relocation-model=pic < %s | FileCheck %s --check-prefix=CHECK-PIC-RV64

define signext i32 @foo(i32 signext %a, i32 signext %b) {
entry:
  ; CHECK-NOPIC-RV32: call  __divsi3
  ; CHECK-PIC-RV32: call  __divsi3@plt
  ; CHECK-NOPIC-RV64: call  __divdi3
  ; CHECK-PIC-RV64: call  __divdi3@plt

  %div = sdiv i32 %a, %b
  ret i32 %div
}

