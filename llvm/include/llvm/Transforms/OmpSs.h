//===-- OmpSs.h - OmpSs Transformations -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the OmpSs transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_OMPSS_H
#define LLVM_TRANSFORMS_OMPSS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class BasicBlockPass;
class Function;
class FunctionPass;
class ModulePass;
class Pass;
class GetElementPtrInst;
class PassInfo;
class TargetLowering;
class TargetMachine;

//===----------------------------------------------------------------------===//
//
// OmpSsPass - Modifies IR in order to translate intrinsics to legal OmpSs-2 code
ModulePass *createOmpSsPass();

struct OmpSsPass : public PassInfoMixin<OmpSsPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // End llvm namespace

#endif
