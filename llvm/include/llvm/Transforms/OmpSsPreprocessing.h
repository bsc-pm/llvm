//===-- OmpSsPreprocessing.h - OmpSs Pre-lowering transformations -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the OmpSs pre-lowering transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_OMPSS_PREPROCESSING_H
#define LLVM_TRANSFORMS_OMPSS_PREPROCESSING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class ModulePass;

//===----------------------------------------------------------------------===//
//
// OmpSsPass - Modifies IR in order to translate intrinsics to legal OmpSs-2 code
ModulePass *createOmpSsPreprocessingPass();

struct OmpSsPreprocessingPass : public PassInfoMixin<OmpSsPreprocessingPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // End llvm namespace

#endif
