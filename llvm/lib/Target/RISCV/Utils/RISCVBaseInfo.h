//===-- RISCVBaseInfo.h - Top level definitions for RISCV MC ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone enum definitions for the RISCV target
// useful for the compiler back-end and the MC libraries.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVBASEINFO_H
#define LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVBASEINFO_H

#include "RISCVRegisterInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/SubtargetFeature.h"

namespace llvm {

// RISCVII - This namespace holds all of the target specific flags that
// instruction info tracks. All definitions must match RISCVInstrFormats.td.
namespace RISCVII {
enum {
  InstFormatPseudo = 0,
  InstFormatR = 1,
  InstFormatR4 = 2,
  InstFormatI = 3,
  InstFormatS = 4,
  InstFormatB = 5,
  InstFormatU = 6,
  InstFormatJ = 7,
  InstFormatCR = 8,
  InstFormatCI = 9,
  InstFormatCSS = 10,
  InstFormatCIW = 11,
  InstFormatCL = 12,
  InstFormatCS = 13,
  InstFormatCA = 14,
  InstFormatCB = 15,
  InstFormatCJ = 16,
  InstFormatOther = 17,

  // EPI formats
  InstEPIVAluInt = 17,
  InstEPIVAluIntImm = 18,
  InstEPIVAluFloat = 19,
  InstEPIVAluFM = 20,
  InstEPIVLoadInt = 21,
  InstEPIVLoadFloat = 22,
  InstEPIVAtomic = 23,
  // End of EPI formats

  InstFormatMask = 31
};

enum {
  MO_None,
  MO_CALL,
  MO_PLT,
  MO_LO,
  MO_HI,
  MO_PCREL_LO,
  MO_PCREL_HI,
  MO_GOT_HI,
  MO_TPREL_LO,
  MO_TPREL_HI,
  MO_TPREL_ADD,
  MO_TLS_GOT_HI,
  MO_TLS_GD_HI,
};

} // namespace RISCVII

namespace EPICSR {

enum {
  VSTART = 0x008,
  VXSAT = 0x009,
  VXRM = 0x00A,
  VL = 0xC20,
  VTYPE = 0xC21,
};

}

namespace RISCVOp {
enum OperandType : unsigned {
  OPERAND_FIRST_RISCV_IMM = MCOI::OPERAND_FIRST_TARGET,
  OPERAND_UIMM4 = OPERAND_FIRST_RISCV_IMM,
  OPERAND_UIMM5,
  OPERAND_UIMM12,
  OPERAND_SIMM12,
  OPERAND_SIMM13_LSB0,
  OPERAND_UIMM20,
  OPERAND_SIMM21_LSB0,
  OPERAND_UIMMLOG2XLEN,
  OPERAND_LAST_RISCV_IMM = OPERAND_UIMMLOG2XLEN
};
} // namespace RISCVOp

// Describes the predecessor/successor bits used in the FENCE instruction.
namespace RISCVFenceField {
enum FenceField {
  I = 8,
  O = 4,
  R = 2,
  W = 1
};
}

// Describes the supported floating point rounding mode encodings.
namespace RISCVFPRndMode {
enum RoundingMode {
  RNE = 0,
  RTZ = 1,
  RDN = 2,
  RUP = 3,
  RMM = 4,
  DYN = 7,
  Invalid
};

inline static StringRef roundingModeToString(RoundingMode RndMode) {
  switch (RndMode) {
  default:
    llvm_unreachable("Unknown floating point rounding mode");
  case RISCVFPRndMode::RNE:
    return "rne";
  case RISCVFPRndMode::RTZ:
    return "rtz";
  case RISCVFPRndMode::RDN:
    return "rdn";
  case RISCVFPRndMode::RUP:
    return "rup";
  case RISCVFPRndMode::RMM:
    return "rmm";
  case RISCVFPRndMode::DYN:
    return "dyn";
  }
}

inline static RoundingMode stringToRoundingMode(StringRef Str) {
  return StringSwitch<RoundingMode>(Str)
      .Case("rne", RISCVFPRndMode::RNE)
      .Case("rtz", RISCVFPRndMode::RTZ)
      .Case("rdn", RISCVFPRndMode::RDN)
      .Case("rup", RISCVFPRndMode::RUP)
      .Case("rmm", RISCVFPRndMode::RMM)
      .Case("dyn", RISCVFPRndMode::DYN)
      .Default(RISCVFPRndMode::Invalid);
}

inline static bool isValidRoundingMode(unsigned Mode) {
  switch (Mode) {
  default:
    return false;
  case RISCVFPRndMode::RNE:
  case RISCVFPRndMode::RTZ:
  case RISCVFPRndMode::RDN:
  case RISCVFPRndMode::RUP:
  case RISCVFPRndMode::RMM:
  case RISCVFPRndMode::DYN:
    return true;
  }
}
} // namespace RISCVFPRndMode

// EPI Vector mask
namespace RISCVEPIVectorMask {

inline static Register stringToVectorMask(StringRef Str) {
  return StringSwitch<Register>(Str)
      .Case("v0.t", RISCV::V0)
      .Default(RISCV::NoRegister);
}

} // namespace RISCVEPIVectorMask

namespace RISCVEPIVectorElementWidth {

#define VECTOR_ELEMENT_WIDTH_LIST                                              \
  VECTOR_ELEMENT_WIDTH(ElementWidth8, 0, "e8")                                     \
  VECTOR_ELEMENT_WIDTH(ElementWidth16, 1, "e16")                                   \
  VECTOR_ELEMENT_WIDTH(ElementWidth32, 2, "e32")                                   \
  VECTOR_ELEMENT_WIDTH(ElementWidth64, 3, "e64")                                   \
  VECTOR_ELEMENT_WIDTH(ElementWidth128, 4, "e128")

enum VectorElementWidth {
#define VECTOR_ELEMENT_WIDTH(ID, ENC, __) ID = ENC,
  VECTOR_ELEMENT_WIDTH_LIST
#undef VECTOR_ELEMENT_WIDTH
      Invalid
};

inline static VectorElementWidth stringToVectorElementWidth(StringRef Str) {
  return StringSwitch<VectorElementWidth>(Str)
#define VECTOR_ELEMENT_WIDTH(ID, _, STR)                                       \
  .Case(STR, RISCVEPIVectorElementWidth::ID)
      VECTOR_ELEMENT_WIDTH_LIST
#undef VECTOR_ELEMENT_WIDTH
          .Default(RISCVEPIVectorElementWidth::Invalid);
}

inline static StringRef VectorElementWidthToString(VectorElementWidth VM) {
  switch (VM) {
  default:
    llvm_unreachable("Invalid vector element width");
#define VECTOR_ELEMENT_WIDTH(ID, _, STR)                                       \
  case RISCVEPIVectorElementWidth::ID:                                         \
    return STR;
    VECTOR_ELEMENT_WIDTH_LIST
#undef VECTOR_ELEMENT_WIDTH
  }
}

inline static bool isValidVectorElementWidth(unsigned Mode) {
  switch (Mode) {
  default:
    return false;
#define VECTOR_ELEMENT_WIDTH(ID, _, STR) case RISCVEPIVectorElementWidth::ID:
    VECTOR_ELEMENT_WIDTH_LIST
#undef VECTOR_ELEMENT_WIDTH
    return true;
  }
}

#undef VECTOR_ELEMENT_WIDTH_LIST

} // namespace RISCVEPIVectorElementWidth

namespace RISCVEPIVectorMultiplier {

#define VECTOR_MULTIPLIER_LIST                                                 \
  VECTOR_MULTIPLIER(VMul1, 0, "m1")                                            \
  VECTOR_MULTIPLIER(VMul2, 1, "m2")                                            \
  VECTOR_MULTIPLIER(VMul4, 2, "m4")                                            \
  VECTOR_MULTIPLIER(VMul8, 3, "m8")

enum VectorMultiplier {
#define VECTOR_MULTIPLIER(ID, ENC, __) ID = ENC,
  VECTOR_MULTIPLIER_LIST
#undef VECTOR_MULTIPLIER
  Invalid
};

inline static VectorMultiplier stringToVectorMultiplier(StringRef Str) {
  return StringSwitch<VectorMultiplier>(Str)
#define VECTOR_MULTIPLIER(ID, _, STR) .Case(STR, RISCVEPIVectorMultiplier::ID)
      VECTOR_MULTIPLIER_LIST
#undef VECTOR_MULTIPLIER
      .Default(RISCVEPIVectorMultiplier::Invalid);
}

inline static StringRef VectorMultiplierToString(VectorMultiplier VM) {
  switch (VM) {
  default:
    llvm_unreachable("Invalid vector type");
#define VECTOR_MULTIPLIER(ID, _, STR) case RISCVEPIVectorMultiplier::ID: return STR;
  VECTOR_MULTIPLIER_LIST
#undef VECTOR_MULTIPLIER
  }
}

inline static bool isValidVectorMultiplier(unsigned Mode) {
  switch (Mode) {
  default:
    return false;
#define VECTOR_MULTIPLIER(ID, _, STR) case RISCVEPIVectorMultiplier::ID:
  VECTOR_MULTIPLIER_LIST
#undef VECTOR_MULTIPLIER
    return true;
  }
}

#undef VECTOR_MULTIPLIER_LIST

} // namespace RISCVEPIVectorMultiplier

namespace RISCVSysReg {
struct SysReg {
  const char *Name;
  unsigned Encoding;
  // FIXME: add these additional fields when needed.
  // Privilege Access: Read, Write, Read-Only.
  // unsigned ReadWrite;
  // Privilege Mode: User, System or Machine.
  // unsigned Mode;
  // Check field name.
  // unsigned Extra;
  // Register number without the privilege bits.
  // unsigned Number;
  FeatureBitset FeaturesRequired;
  bool isRV32Only;

  bool haveRequiredFeatures(FeatureBitset ActiveFeatures) const {
    // Not in 32-bit mode.
    if (isRV32Only && ActiveFeatures[RISCV::Feature64Bit])
      return false;
    // No required feature associated with the system register.
    if (FeaturesRequired.none())
      return true;
    return (FeaturesRequired & ActiveFeatures) == FeaturesRequired;
  }
};

#define GET_SysRegsList_DECL
#include "RISCVGenSearchableTables.inc"


} // end namespace RISCVSysReg

namespace RISCVABI {

enum ABI {
  ABI_ILP32,
  ABI_ILP32F,
  ABI_ILP32D,
  ABI_ILP32E,
  ABI_LP64,
  ABI_LP64F,
  ABI_LP64D,
  ABI_Unknown
};

// Returns the target ABI, or else a StringError if the requested ABIName is
// not supported for the given TT and FeatureBits combination.
ABI computeTargetABI(const Triple &TT, FeatureBitset FeatureBits,
                     StringRef ABIName);

// Returns the register used to hold the stack pointer after realignment.
Register getBPReg();

} // namespace RISCVABI

namespace RISCVFeatures {

// Validates if the given combination of features are valid for the target
// triple. Exits with report_fatal_error if not.
void validate(const Triple &TT, const FeatureBitset &FeatureBits);

} // namespace RISCVFeatures

namespace RISCVEPIIntrinsicsTable {

struct EPIIntrinsicInfo {
  unsigned int IntrinsicID;
  unsigned int ClassID;
  unsigned int ExtendedOperand;
  unsigned int MaskOperand;
  unsigned int GVLOperand;
};

#define GET_EPIIntrClassID_DECL
#define GET_EPIIntrinsicsTable_DECL
using namespace RISCV;
#include "RISCVGenSearchableTables.inc"

} // end namespace RISCVEPIIntrinsicsTable

namespace RISCVEPIPseudosTable {

struct EPIPseudoInfo {
  unsigned int Pseudo;
  unsigned int BaseInstr;
  uint8_t VLIndex;
  uint8_t SEWIndex;
  uint8_t MergeOpIndex;
  uint8_t VLMul;

  int getVLIndex() const { return static_cast<int8_t>(VLIndex); }

  int getSEWIndex() const { return static_cast<int8_t>(SEWIndex); }

  int getMergeOpIndex() const { return static_cast<int8_t>(MergeOpIndex); }
};

#define GET_EPIPseudosTable_DECL
using namespace RISCV;
#include "RISCVGenSearchableTables.inc"

} // end namespace RISCVEPIPseudosTable

} // namespace llvm

#endif
