//===-- RISCVRegisterInfo.td - RISC-V EPI Register defs ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def EPIFPR : RegisterClass<"RISCV", [f32, f64], 64, (add
    (sequence "F%u_64", 0, 7),
    (sequence "F%u_64", 10, 17),
    (sequence "F%u_64", 28, 31),
    (sequence "F%u_64", 8, 9),
    (sequence "F%u_64", 18, 27)
)>;

// FIXME: In many places we use 128 but there isn't actually a minimum size
// defined, so this must be generalised somehow.

// Define V0-V31
let RegAltNameIndices = [ABIRegAltName] in {
   foreach Index = 0-31 in {
      def V#Index : RISCVReg<Index, "v"#Index, ["v"#Index]>;
   }
}

def EPIVR : RegisterClass<"RISCV",
                          [nxv1i64, nxv1i32, nxv1i16, nxv1i8, nxv1i1,
                           nxv1f64, nxv1f32],
                          128,
                          (add (sequence "V%u", 0, 31))> {
  let isDynamicSpillSize = 1;
}

// Represents the current vector length
def VL : RISCVReg<0, "vl">;
def VLR : RegisterClass<"RISCV", [XLenVT], 32, (add VL)> {
  let CopyCost = -1; // Don't allow copying
  let isAllocatable = 0; // Not allocatable
  let RegInfos = RegInfoByHwMode<
      [RV32,              RV64,              DefaultMode],
      [RegInfo<32,32,32>, RegInfo<64,64,64>, RegInfo<32,32,32>]>;
}

// Represents the current vector type. This has been decoupled because
// some instructions are not polymorphic so they don't use VTYPE.
def VTYPE : RISCVReg<0, "vtype">;
def VTypeR : RegisterClass<"RISCV", [XLenVT], 32, (add VTYPE)> {
  let CopyCost = -1; // Don't allow copying
  let isAllocatable = 0; // Not allocatable
  let RegInfos = RegInfoByHwMode<
      [RV32,              RV64,              DefaultMode],
      [RegInfo<32,32,32>, RegInfo<64,64,64>, RegInfo<32,32,32>]>;
}
