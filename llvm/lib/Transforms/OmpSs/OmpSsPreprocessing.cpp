//===- OmpSsPreprocessing.cpp -- Strip parts of Debug Info --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/OmpSsPreprocessing.h"

#include "llvm/InitializePasses.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

namespace {
struct OmpSsPreprocessingModule {
  Module &M;
  LLVMContext &Ctx;
  OmpSsPreprocessingModule(Module &M)
      : M(M), Ctx(M.getContext())
        {}

  bool run() {
    if (M.empty())
      return false;

    bool Modified = false;
    for (auto &F : M)
      if (!F.isDeclaration())
        Modified = removeUnreachableBlocks(
          F, /*DTU=*/nullptr, /*MSSAU=*/nullptr,
          /*UnreachNotReturnF=*/false);
    return Modified;
  }
};

struct OmpSsPreprocessingLegacyPass : public ModulePass {
  /// Pass identification, replacement for typeid
  static char ID;
  OmpSsPreprocessingLegacyPass() : ModulePass(ID) {
    initializeOmpSsPreprocessingLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;
    return OmpSsPreprocessingModule(M).run();
  }

  StringRef getPassName() const override { return "OmpSs-2 CFG pre-lowering transforms"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override { }
};

} // end namespace

PreservedAnalyses OmpSsPreprocessingPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!OmpSsPreprocessingModule(M).run())
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

char OmpSsPreprocessingLegacyPass::ID = 0;

ModulePass *llvm::createOmpSsPreprocessingPass() {
  return new OmpSsPreprocessingLegacyPass();
}

// void LLVMOmpSsPreprocessingPass(LLVMPassManagerRef PM) {
//   unwrap(PM)->add(createOmpSsPreprocessingPass());
// }

INITIALIZE_PASS_BEGIN(OmpSsPreprocessingLegacyPass, "ompss-2-pre",
                "Transforms applies necessary CFG tranformation before running the transformation pass", false, false)
INITIALIZE_PASS_END(OmpSsPreprocessingLegacyPass, "ompss-2-pre",
                "Transforms applies necessary CFG tranformation before running the transformation pass", false, false)
