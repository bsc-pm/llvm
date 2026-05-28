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
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsOmpSs.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

namespace {

// True if BB holds an OmpSs directive intrinsic that must survive
// CFG cleanup.
static bool containsOmpSsDirectiveIntrinsic(const BasicBlock &BB) {
  for (const Instruction &I : BB) {
    if (const auto *II = dyn_cast<IntrinsicInst>(&I)) {
      Intrinsic::ID ID = II->getIntrinsicID();
      if (ID == Intrinsic::directive_region_entry ||
          ID == Intrinsic::directive_region_exit ||
          ID == Intrinsic::directive_marker)
        return true;
    }
  }
  return false;
}

// If a task body cannot reach its end, Clang emits the matching
// directive.region.exit in an unreachable block. Hook those blocks as
// (runtime-unreachable) cases of a synthetic switch at the function entry
// so removeUnreachableBlocks keeps them, preserving the entry/exit pair.
static bool preserveDirectiveBlocks(Function &F) {
  SmallPtrSet<BasicBlock *, 8> Reachable;
  SmallVector<BasicBlock *, 8> Worklist;
  BasicBlock &EntryBB = F.getEntryBlock();
  Worklist.push_back(&EntryBB);
  Reachable.insert(&EntryBB);
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();
    for (BasicBlock *Succ : successors(BB))
      if (Reachable.insert(Succ).second)
        Worklist.push_back(Succ);
  }

  SmallVector<BasicBlock *, 4> ToHook;
  for (BasicBlock &BB : F)
    if (!Reachable.count(&BB) && containsOmpSsDirectiveIntrinsic(BB))
      ToHook.push_back(&BB);

  if (ToHook.empty())
    return false;

  // `freeze i32 undef` as the switch condition: not a Constant (so
  // ConstantFoldTerminator can't collapse the switch) and has only a
  // Constant operand (so OmpSsRegionAnalysis's bundle check is happy).
  Instruction *Term = EntryBB.getTerminator();
  assert(Term->getNumSuccessors() == 1 &&
         "Expected entry to fall through to a single successor");
  BasicBlock *OrigSucc = Term->getSuccessor(0);

  IRBuilder<> B(Term);
  llvm::IntegerType *I32Ty = B.getInt32Ty();
  Value *Cond = B.CreateFreeze(UndefValue::get(I32Ty), "ompss.dir.anchor");
  SwitchInst *SI = B.CreateSwitch(Cond, OrigSucc, ToHook.size());
  unsigned CaseVal = 1;
  for (BasicBlock *BB : ToHook)
    SI->addCase(ConstantInt::get(I32Ty, CaseVal++), BB);
  Term->eraseFromParent();
  return true;
}

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
      if (!F.isDeclaration()) {
        Modified |= preserveDirectiveBlocks(F);
        Modified |= removeUnreachableBlocks(
          F, /*DTU=*/nullptr, /*MSSAU=*/nullptr,
          /*UnreachNotReturnF=*/false);
      }
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
