; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %s \
; RUN:   | FileCheck --check-prefix=RV32 %s
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s \
; RUN:   | FileCheck --check-prefix=RV64 %s

; This test ensures that the 'fp' register is defined as a CFA whenever the
; frame pointer is used.

define void @test_fp_as_cfa(i32 signext %val) {
; RV32-LABEL: test_fp_as_cfa:
; RV32:         .cfi_def_cfa s0, 0
; RV32-NEXT:    .cfi_offset ra, -4
; RV32-NEXT:    .cfi_offset s0, -8

; RV64-LABEL: test_fp_as_cfa:
; RV64:         .cfi_def_cfa s0, 0
; RV64-NEXT:    .cfi_offset ra, -8
; RV64-NEXT:    .cfi_offset s0, -16
    %1 = alloca i32, i32 %val
    ret void
}
