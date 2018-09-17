; RUN: llc -mtriple=riscv32 < %s | FileCheck --check-prefix=RV32 %s
; RUN: llc -mtriple=riscv64 < %s | FileCheck --check-prefix=RV64 %s

define dso_local void @realign() {
; RV32-LABEL: realign:
; RV32:    .cfi_def_cfa 8, 0
; RV32-NEXT:    .cfi_offset 1, -4
; RV32-NEXT:    .cfi_offset 8, -8

; RV64-LABEL: realign:
; RV64:    .cfi_def_cfa 8, 0
; RV64-NEXT:    .cfi_offset 1, -8
; RV64-NEXT:    .cfi_offset 8, -16
entry:
  %x = alloca i32, align 32
  %x.0.x.0. = load volatile i32, i32* %x, align 32
  %inc = add nsw i32 %x.0.x.0., 1
  store volatile i32 %inc, i32* %x, align 32
  ret void
}

define dso_local void @no_realign() {
; RV32-LABEL: no_realign:
; RV32:    .cfi_def_cfa_offset 16

; RV64-LABEL: no_realign:
; RV64:    .cfi_def_cfa_offset 16
entry:
  %x = alloca i32, align 4
  %x.0.x.0. = load volatile i32, i32* %x, align 4
  %inc = add nsw i32 %x.0.x.0., 1
  store volatile i32 %inc, i32* %x, align 4
  ret void
}
