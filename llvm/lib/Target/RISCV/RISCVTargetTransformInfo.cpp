//===-- RISCVTargetTransformInfo.cpp - RISC-V specific TTI ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVTargetTransformInfo.h"
#include "Utils/RISCVMatInt.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
using namespace llvm;

#define DEBUG_TYPE "riscvtti"

// TODO: VK june 27, 18:15
// This method calls `getIntMatCost` which is defined in upstream master commit
// # 9f155bc6e592f7491eeb8af90558ad9048d5c988. Patching this commit might cause
// more than simple conflicts with changes in the EPI branch. We don't really
// need it as of now, so it's easier to just comment out the main logic and just
// return a zero always. We can change it before EPI rebase.
//
int RISCVTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  return TTI::TCC_Free;
  /*
   *
   *  // We have a Zero register, so 0 is always free.
   *  if (Imm == 0)
   *    return TTI::TCC_Free;
   *
   *  // Otherwise, we check how many instructions it will take to materialise.
   *  const DataLayout &DL = getDataLayout();
   *  return RISCVMatInt::getIntMatCost(Imm, DL.getTypeSizeInBits(Ty),
   *                                    getST()->is64Bit());
   */
}

int RISCVTTIImpl::getIntImmCost(unsigned Opcode, unsigned Idx, const APInt &Imm,
                                Type *Ty) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Some instructions in RISC-V can take a 12-bit immediate. Some of these are
  // commutative, in others the immediate comes from a specific argument index.
  bool Takes12BitImm = false;
  unsigned ImmArgIdx = ~0U;

  switch (Opcode) {
  case Instruction::GetElementPtr:
    // Never hoist any arguments to a GetElementPtr. CodeGenPrepare will
    // split up large offsets in GEP into better parts than ConstantHoisting
    // can.
    return TTI::TCC_Free;
  case Instruction::Add:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Mul:
    Takes12BitImm = true;
    break;
  case Instruction::Sub:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    Takes12BitImm = true;
    ImmArgIdx = 1;
    break;
  default:
    break;
  }

  if (Takes12BitImm) {
    // Check immediate is the correct argument...
    if (Instruction::isCommutative(Opcode) || Idx == ImmArgIdx) {
      // ... and fits into the 12-bit immediate.
      if (Imm.getMinSignedBits() <= 64 &&
          getTLI()->isLegalAddImmediate(Imm.getSExtValue())) {
        return TTI::TCC_Free;
      }
    }

    // Otherwise, use the full materialisation cost.
    return getIntImmCost(Imm, Ty);
  }

  // By default, prevent hoisting.
  return TTI::TCC_Free;
}

int RISCVTTIImpl::getIntImmCost(Intrinsic::ID IID, unsigned Idx,
                                const APInt &Imm, Type *Ty) {
  // Prevent hoisting in unknown cases.
  return TTI::TCC_Free;
}

unsigned RISCVTTIImpl::getNumberOfRegisters(bool Vector) {
  // FIXME: VK Jun 28
  // For the time being we just return a fixed 32 number of vector registers.
  //
  return Vector ? 32 : 1;
}
