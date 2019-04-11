//===-- RISCVRegisterInfo.td - RISC-V EPI Register defs ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// FIXME: In many places we use 128 but there isn't actually a minimum size
// defined, so this must be generalised somehow.

// Define V0-V31
let RegAltNameIndices = [ABIRegAltName] in {
   foreach Index = 0-31 in {
      def V#Index : RISCVReg<Index, "v"#Index, ["v"#Index]>;
   }
}

let Namespace = "RISCV" in {
def epivrodd : SubRegIndex<0, -1>;
def epivreven : SubRegIndex<0, -1>;

def epivr2odd : SubRegIndex<0, -1>;
def epivr2even : SubRegIndex<0, -1>;

def epivr4odd : SubRegIndex<0, -1>;
def epivr4even : SubRegIndex<0, -1>;

let SubRegIndices = [epivreven, epivrodd] in
{
  foreach Index = 0-15 in {
    def "V"#!shl(#Index, 1)#"_2" : RegisterWithSubRegs<"",
      [!cast<Register>("V"#!shl(#Index, 1)),
       !cast<Register>("V"#!add(!shl(#Index, 1), 1))]>;
  }
}

let SubRegIndices = [epivr2even, epivr2odd] in
{
  foreach Index = 0-7 in {
    def "V"#!shl(#Index, 2)#"_4" : RegisterWithSubRegs<"",
      [!cast<Register>("V"#!shl(#Index, 2)#"_2"),
       !cast<Register>("V"#!add(!shl(#Index, 2), 2)#"_2")]>;
  }
}

let SubRegIndices = [epivr4even, epivr4odd] in
{
  foreach Index = 0-3 in {
    def "V"#!shl(#Index, 3)#"_8" : RegisterWithSubRegs<"",
      [!cast<Register>("V"#!shl(#Index, 3)#"_4"),
       !cast<Register>("V"#!add(!shl(#Index, 3), 4)#"_4")]>;
  }
}
} // Namespace = "RISCV"

class EPIVReg<list<ValueType> regTypes, dag regList, int Vlmul>
  : RegisterClass<"RISCV",
                  regTypes,
                  128,
                  regList> {
  int VLMul = Vlmul;
  let isDynamicSpillSize = 1;
}

def EPIVR : EPIVReg<[nxv1i64, nxv2i32, nxv4i16, nxv8i8, nxv1f64, nxv2f32,
                     nxv1i1, nxv2i1, nxv4i1, nxv8i1, nxv16i1, nxv32i1 /* FIXME, nxv64i1 doesn't exist yet */],
                    // Prioritise caller-save and then callee-save
                    (add (sequence "V%u", 0, 7),
                         (sequence "V%u", 16, 23),
                         (sequence "V%u", 8, 15),
                         (sequence "V%u", 24, 31)),
                    1>;

def EPIVR2 : EPIVReg<[nxv2i64, nxv4i32, nxv8i16, nxv16i8, nxv2f64, nxv4f32],
                     // Prioritise caller-save and then callee-save
                     (add (sequence "V%u_2", 0, 7, 2),
                          (sequence "V%u_2", 16, 23, 2),
                          (sequence "V%u_2", 8, 15, 2),
                          (sequence "V%u_2", 24, 31, 2)),
                     2>;

def EPIVR4 : EPIVReg<[nxv4i64, nxv8i32, nxv16i16, nxv32i8, nxv4f64, nxv8f32],
                     // Prioritise caller-save and then callee-save
                     (add (sequence "V%u_4", 0, 7, 4),
                          (sequence "V%u_4", 16, 23, 4),
                          (sequence "V%u_4", 8, 15, 4),
                          (sequence "V%u_4", 24, 31, 4)),
                     4>;

def EPIVR8 : EPIVReg<[nxv8i64, nxv16i32, nxv32i16, /*FIXME nxv64i8 does not exist yet,*/ nxv8f64, nxv16f32],
                     // Prioritise caller-save and then callee-save
                     (add V0_8, V16_8, V8_8, V24_8),
                     8>;

class EPIVRegList<list<EPIVReg> Evr>
{
  list<EPIVReg> Values = Evr;
}

def AllEPIVRegs
    : EPIVRegList<[EPIVR, EPIVR2, EPIVR4, EPIVR8]>;

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
