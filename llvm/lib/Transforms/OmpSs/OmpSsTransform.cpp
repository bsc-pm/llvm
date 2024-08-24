//===- OmpSs.cpp -- Strip parts of Debug Info --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/OmpSs.h"
#include "llvm/Transforms/OmpSs/Nanos6API.h"
#include "llvm/Analysis/OmpSsRegionAnalysis.h"

#include "llvm/InitializePasses.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IntrinsicsOmpSs.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
using namespace llvm;

namespace {

struct DirectiveFinalInfo {
  SmallVector<Instruction *> InnerClonedEntries;
  SmallVector<Instruction *> InnerClonedExits;
  // Used to get final cloned instructions except for
  // inner directives entry/exit since they will be removed
  // before the current lowering directive.
  ValueToValueMapTy VMap;
};

static void registerCheckVersion(Module &M) {
  Function *Func = cast<Function>(nanos6Api::registerCtorCheckVersionFuncCallee(M).getCallee());
  if (Func->empty()) {
    Func->setLinkage(GlobalValue::InternalLinkage);
    BasicBlock *EntryBB = BasicBlock::Create(M.getContext(), "entry", Func);
    Instruction *RetInst = ReturnInst::Create(M.getContext());
    RetInst->insertInto(EntryBB, EntryBB->end());

    appendToGlobalCtors(M, Func, 65535);

    BasicBlock &Entry = Func->getEntryBlock();

    // struct nanos6_version_t {
    //   uint64_t family;
    //   uint64_t majorversion;
    //   uint64_t minor_version;
    // };

    Type *Int64Ty = Type::getInt64Ty(M.getContext());

    SmallVector<Constant *, 4> Versions;
    const size_t NUM_VERSIONS = 1;
    const size_t TUPLE_SIZE = 3;
    const size_t TOTAL_ELEMS = TUPLE_SIZE*NUM_VERSIONS;
    Constant *NumVersionsValue = ConstantInt::get(Int64Ty, NUM_VERSIONS);
    // family
    Versions.push_back(
      ConstantInt::get(Int64Ty, 0));
    // major_version
    Versions.push_back(
      ConstantInt::get(Int64Ty, 1));
    // minor_version
    Versions.push_back(
      ConstantInt::get(Int64Ty, 0));

    GlobalVariable *VersionListValue =
      new GlobalVariable(M, ArrayType::get(Int64Ty, TOTAL_ELEMS),
        /*isConstant=*/true, GlobalVariable::InternalLinkage,
        ConstantArray::get(ArrayType::get(Int64Ty, TOTAL_ELEMS),
          Versions), "nanos6_versions");

    IRBuilder<> BBBuilder(&Entry.back());
    Constant *SourceFilenameGV = BBBuilder.CreateGlobalStringPtr(M.getSourceFileName());
    BBBuilder.CreateCall(
      nanos6Api::checkVersionFuncCallee(M), {NumVersionsValue, VersionListValue, SourceFilenameGV});
  }
}

struct OmpSsDirective {
  Module &M;
  LLVMContext &Ctx;
  const DataLayout &DL;
  Function &F;
  const DirectiveInfo &DirInfo;
  const DirectiveEnvironment &DirEnv;
  const DirectiveDSAInfo &DSAInfo;
  const DirectiveVLADimsInfo &VLADimsInfo;
  const DirectiveDependsInfo &DependsInfo;
  const DirectiveReductionsInitCombInfo &ReductionsInitCombInfo;
  const DirectiveCostInfo &CostInfo;
  const DirectivePriorityInfo &PriorityInfo;
  const DirectiveOnreadyInfo &OnreadyInfo;
  const DirectiveDeviceInfo &DeviceInfo;
  const DirectiveCapturedInfo &CapturedInfo;
  const DirectiveNonPODsInfo &NonPODsInfo;
  const DirectiveLoopInfo &LoopInfo;
  const DirectiveWhileInfo &WhileInfo;
  DirectiveFinalInfo &FinalInfo;
  Type *Int32Ty;
  Type *Int8Ty;
  Type *Int64Ty;
  PointerType *PtrTy;
  bool HasDevice;

  // 6 for ndrange and 1 for shm_size
  const size_t DeviceArgsSize = 7;
  nanos6Api::Nanos6MultidepFactory MultidepFactory;
  // This is a hack to move some instructions to its corresponding functions
  // at the end of the pass.
  // For example, this is used to move allocas to its corresponding
  // function entry.
  SmallVectorImpl<Instruction *> &PostMoveInstructions;

  /// InFinalRAIIObject - This sets the OmpSsDirective::InFinalCtx bool and
  /// restores it when destroyed.  This says that in the current lowering task
  /// we are in a final context so we have to set if(0) flag by hand.
  /// NOTE: this only affects device ndrange tasks
  class InFinalRAIIObject {
    OmpSsDirective &P;
    bool OldVal;
  public:
    InFinalRAIIObject(OmpSsDirective &p, bool Value = true)
      : P(p), OldVal(P.InFinalCtx) {
      P.InFinalCtx = Value;
    }

    /// restore - This can be used to restore the state early, before the dtor
    /// is run.
    void restore() {
      P.InFinalCtx = OldVal;
    }

    ~InFinalRAIIObject() {
      restore();
    }
  };
  bool InFinalCtx = false;

  OmpSsDirective(Module &M, Function &F, DirectiveInfo &DirInfo,
                 DirectiveFinalInfo &FinalInfo,
                 SmallVectorImpl<Instruction *> &PostMoveInstructions)
      : M(M), Ctx(M.getContext()), DL(M.getDataLayout()),
        F(F), DirInfo(DirInfo),
        DirEnv(DirInfo.DirEnv),
        DSAInfo(DirEnv.DSAInfo),
        VLADimsInfo(DirEnv.VLADimsInfo),
        DependsInfo(DirEnv.DependsInfo),
        ReductionsInitCombInfo(DirEnv.ReductionsInitCombInfo),
        CostInfo(DirEnv.CostInfo),
        PriorityInfo(DirEnv.PriorityInfo),
        OnreadyInfo(DirEnv.OnreadyInfo),
        DeviceInfo(DirEnv.DeviceInfo),
        CapturedInfo(DirEnv.CapturedInfo),
        NonPODsInfo(DirEnv.NonPODsInfo),
        LoopInfo(DirEnv.LoopInfo),
        WhileInfo(DirEnv.WhileInfo),
        FinalInfo(FinalInfo),
        Int32Ty(Type::getInt32Ty(Ctx)), Int8Ty(Type::getInt8Ty(Ctx)),
        Int64Ty(Type::getInt64Ty(Ctx)), PtrTy(PointerType::getUnqual(Ctx)),
        HasDevice(!DeviceInfo.empty()),
        PostMoveInstructions(PostMoveInstructions)
        {}

  // For each iterator compute the normalized bounds [0, ub) handling
  // outer iterator usage in the current one
  void ComputeLoopBounds(Instruction *InsertPt, SmallVectorImpl<Value *> &UBounds) {
    IRBuilder<> IRB(InsertPt);
    for (size_t i = 0; i < LoopInfo.LBound.size(); ++i) {
      // Set ub/lb of referenced iterators based on the loop type
      // for (int i = 0; i < 100; ++i)
      //   for (int j = i; j > 0; --j)  replaces i by ub(i)-1

      // First determite the pessimistic lb/ub. A pessimistic bound
      // is the lower/upper executable iteration. That is,
      // in a '<' the upper bound will be ubound - 1, whereas
      // in a '<= the upper bound will be ubound.
      SmallVector<Value *> PessimistLB(i);
      SmallVector<Value *> PessimistUB(i);
      for (size_t j = 0; j < i; ++j) {
        switch (LoopInfo.LoopType[j]) {
        case DirectiveLoopInfo::LE:
          PessimistLB[j] = LoopInfo.LBound[j].Result;
          PessimistUB[j] = LoopInfo.UBound[j].Result;
          break;
        case DirectiveLoopInfo::LT:
          PessimistLB[j] = LoopInfo.LBound[j].Result;
          PessimistUB[j] =
            IRB.CreateSub(
              LoopInfo.UBound[j].Result,
              ConstantInt::get(
                LoopInfo.UBound[j].Result->getType(), 1));
          break;
        case DirectiveLoopInfo::GT:
          PessimistLB[j] =
            IRB.CreateAdd(
              LoopInfo.UBound[j].Result,
              ConstantInt::get(
                LoopInfo.UBound[j].Result->getType(), 1));
          PessimistUB[j] = LoopInfo.LBound[j].Result;
          break;
        case DirectiveLoopInfo::GE:
          PessimistLB[j] = LoopInfo.UBound[j].Result;
          PessimistUB[j] = LoopInfo.LBound[j].Result;
          break;
        default:
          llvm_unreachable("unexpected loop type");
        }
      }
      // Once we have the pessimistic bounds compute the current
      // bounds based on the loop type. That is, in a '>' the lower bound
      // will be the pessimistic ubound
      if (LoopInfo.LoopType[i] == DirectiveLoopInfo::GT
          || LoopInfo.LoopType[i] == DirectiveLoopInfo::GE) {
        for (size_t j = 0; j < i; ++j)
          IRB.CreateStore(PessimistUB[j], LoopInfo.IndVar[j]);
        LoopInfo.LBound[i].Result = IRB.CreateCall(LoopInfo.LBound[i].Fun, LoopInfo.LBound[i].Args);
        for (size_t j = 0; j < i; ++j)
          IRB.CreateStore(PessimistLB[j], LoopInfo.IndVar[j]);
        LoopInfo.UBound[i].Result = IRB.CreateCall(LoopInfo.UBound[i].Fun, LoopInfo.UBound[i].Args);
      } else {
        for (size_t j = 0; j < i; ++j)
          IRB.CreateStore(PessimistLB[j], LoopInfo.IndVar[j]);
        LoopInfo.LBound[i].Result = IRB.CreateCall(LoopInfo.LBound[i].Fun, LoopInfo.LBound[i].Args);
        for (size_t j = 0; j < i; ++j)
          IRB.CreateStore(PessimistUB[j], LoopInfo.IndVar[j]);
        LoopInfo.UBound[i].Result = IRB.CreateCall(LoopInfo.UBound[i].Fun, LoopInfo.UBound[i].Args);
      }

      // Step is not allowed to depend on other iterators
      LoopInfo.Step[i].Result = IRB.CreateCall(LoopInfo.Step[i].Fun, LoopInfo.Step[i].Args);

      // <      0, (ub - 1 - lb) / step + 1
      // <=     0, (ub - lb)     / step + 1
      // >      0, (ub + 1 - lb) / step + 1
      // >=     0, (ub - lb)     / step + 1
      auto p = buildSubSignDependent(
        IRB, LoopInfo.UBound[i].Result, LoopInfo.LBound[i].Result, LoopInfo.UBoundSigned[i], LoopInfo.LBoundSigned[i]);
      Value *RegisterUpperB = p.first;

      switch (LoopInfo.LoopType[i]) {
      case DirectiveLoopInfo::LT:
        RegisterUpperB = IRB.CreateSub(RegisterUpperB, ConstantInt::get(RegisterUpperB->getType(), 1));
        break;
      case DirectiveLoopInfo::GT:
        RegisterUpperB = IRB.CreateAdd(RegisterUpperB, ConstantInt::get(RegisterUpperB->getType(), 1));
        break;
      case DirectiveLoopInfo::LE:
      case DirectiveLoopInfo::GE:
        break;
      default:
        llvm_unreachable("unexpected loop type");
      }
      p = buildDivSignDependent(
        IRB, RegisterUpperB, LoopInfo.Step[i].Result, p.second, LoopInfo.Step[i].Result);
      RegisterUpperB = IRB.CreateAdd(p.first, ConstantInt::get(p.first->getType(), 1));

      RegisterUpperB =
        createZSExtOrTrunc(
          IRB, RegisterUpperB,
          nanos6Api::Nanos6LoopBounds::getInstance(M).getUBType(), p.second);

      UBounds[i] = RegisterUpperB;
    }
  }

  // Signed extension of V to type Ty
  static Value *createZSExtOrTrunc(IRBuilder<> &IRB, Value *V, Type *Ty, bool Signed) {
    if (Signed)
      return IRB.CreateSExtOrTrunc(V, Ty);
    return IRB.CreateZExtOrTrunc(V, Ty);
  }

  // Compares LHS and RHS and extends the one with lower type size. The extension
  // is based on LHSSigned/RHSSigned
  // returns the new instruction built and the signedness
  std::pair<Value *, bool> buildInstructionSignDependent(
      IRBuilder<> &IRB,
      Value *LHS, Value *RHS, bool LHSSigned, bool RHSSigned,
      const llvm::function_ref<Value *(IRBuilder<> &, Value *, Value *, bool)> InstrGen) {
    Type *LHSTy = LHS->getType();
    Type *RHSTy = RHS->getType();
    TypeSize LHSTySize = DL.getTypeSizeInBits(LHSTy);
    TypeSize RHSTySize = DL.getTypeSizeInBits(RHSTy);
    // same size LHS and RHS build signed intr. only if both are signed
    bool NewOpSigned = LHSSigned & RHSSigned;
    if (LHSTySize < RHSTySize) {
      NewOpSigned = RHSSigned;
      LHS = createZSExtOrTrunc(IRB, LHS, RHSTy, LHSSigned);
    } else if (LHSTySize > RHSTySize) {
      NewOpSigned = LHSSigned;
      RHS = createZSExtOrTrunc(IRB, RHS, LHSTy, RHSSigned);
    }
    return std::make_pair(InstrGen(IRB, LHS, RHS, NewOpSigned), NewOpSigned);
  }

  std::pair<Value *, bool> buildDivSignDependent(
      IRBuilder<> &IRB,
      Value *LHS, Value *RHS, bool LHSSigned, bool RHSSigned) {
    return buildInstructionSignDependent(
      IRB, LHS, RHS, LHSSigned, RHSSigned,
      [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
        if (NewOpSigned)
          return IRB.CreateSDiv(LHS, RHS);
        return IRB.CreateUDiv(LHS, RHS);
      });
  }

  std::pair<Value *, bool> buildMulSignDependent(
      IRBuilder<> &IRB,
      Value *LHS, Value *RHS, bool LHSSigned, bool RHSSigned) {
    return buildInstructionSignDependent(
      IRB, LHS, RHS, LHSSigned, RHSSigned,
      [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
        return IRB.CreateMul(LHS, RHS);
      });
  }

  std::pair<Value *, bool> buildCmpSignDependent(
      unsigned CmpType, IRBuilder<> &IRB,
      Value *LHS, Value *RHS, bool LHSSigned, bool RHSSigned) {
    return buildInstructionSignDependent(
      IRB, LHS, RHS, LHSSigned, RHSSigned,
      [CmpType](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
        switch (CmpType) {
        case DirectiveLoopInfo::LT:
          if (NewOpSigned)
            return IRB.CreateICmpSLT(LHS, RHS);
          return IRB.CreateICmpULT(LHS, RHS);
        case DirectiveLoopInfo::LE:
          if (NewOpSigned)
            return IRB.CreateICmpSLE(LHS, RHS);
          return IRB.CreateICmpULE(LHS, RHS);
        case DirectiveLoopInfo::GT:
          if (NewOpSigned)
            return IRB.CreateICmpSGT(LHS, RHS);
          return IRB.CreateICmpUGT(LHS, RHS);
        case DirectiveLoopInfo::GE:
          if (NewOpSigned)
            return IRB.CreateICmpSGE(LHS, RHS);
          return IRB.CreateICmpUGE(LHS, RHS);
        }
        llvm_unreachable("unexpected loop type");
      });
  }

  std::pair<Value *, bool> buildAddSignDependent(
      IRBuilder<> &IRB,
      Value *LHS, Value *RHS, bool LHSSigned, bool RHSSigned) {
    return buildInstructionSignDependent(
      IRB, LHS, RHS, LHSSigned, RHSSigned,
      [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
        return IRB.CreateAdd(LHS, RHS);
      });
  }

  std::pair<Value *, bool> buildSubSignDependent(
      IRBuilder<> &IRB,
      Value *LHS, Value *RHS, bool LHSSigned, bool RHSSigned) {
    return buildInstructionSignDependent(
      IRB, LHS, RHS, LHSSigned, RHSSigned,
      [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
        return IRB.CreateSub(LHS, RHS);
      });
  }

  static bool isReplaceableValue(Value *V) {
    return isa<Instruction>(V) || isa<Argument>(V) || isa<GlobalValue>(V);
  }

  static void rewriteUsesInBlocksWithPred(
      Value *Orig, Value *New, llvm::function_ref<bool(Instruction *)> Pred) {

    std::vector<User *> Users(Orig->user_begin(), Orig->user_end());
    for (User *use : Users)
      if (Instruction *inst = dyn_cast<Instruction>(use))
        if (Pred(inst))
          inst->replaceUsesOfWith(Orig, New);
  }

  // Converts all ConstantExpr users of GV in Blocks to instructions
  static void constantExprToInstruction(
      GlobalValue *GV, const SetVector<BasicBlock *> &Blocks) {
    SmallVector<Constant*,4> UsersStack;
    SmallVector<Constant*,4> Worklist;
    Worklist.push_back(GV);
    while (!Worklist.empty()) {
      Constant *C = Worklist.pop_back_val();
      for (auto *U : C->users()) {
        if (isa<ConstantExpr>(U) || isa<ConstantAggregate>(U)) {
          UsersStack.insert(UsersStack.begin(), cast<Constant>(U));
          Worklist.push_back(cast<Constant>(U));
        }
      }
    }

    SmallVector<Value*,4> UUsers;
    for (auto *U : UsersStack) {
      UUsers.clear();
      append_range(UUsers, U->users());
      for (auto *UU : UUsers) {
        if (Instruction *UI = dyn_cast<Instruction>(UU)) {
          if (Blocks.count(UI->getParent())) {
            if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
              Instruction *NewU = CE->getAsInstruction();
              NewU->insertBefore(UI);
              UI->replaceUsesOfWith(U, NewU);
            } else if (ConstantAggregate *CE = dyn_cast<ConstantAggregate>(U)) {
              // Convert the whole constant to a set of InsertValueInst
              Value *NewU = UndefValue::get(CE->getType());
              unsigned Idx = 0;
              while (auto Elt = CE->getAggregateElement(Idx)) {
                NewU = InsertValueInst::Create(NewU, Elt, Idx, "", UI);
                ++Idx;
              }
              UI->replaceUsesOfWith(U, NewU);
            }
          }
        }
      }
    }
  }

  // Build a loop containing Single Entry Single Exit region Entry/Exit
  // and returns the LoopEntry and LoopExit
  void buildLoopForTaskImpl(Type *IndVarTy,
                        Instruction *Entry, Instruction *Exit,
                        const DirectiveLoopInfo &LoopInfo,
                        Instruction *&LoopEntryI, Instruction *&LoopExitI,
                        SmallVectorImpl<Instruction *> &CollapseIterBB,
                        size_t LInfoIndex = 0) {

    IRBuilder<> IRB(Entry);
    IRB.CreateStore(createZSExtOrTrunc(IRB, LoopInfo.LBound[LInfoIndex].Result, IndVarTy, LoopInfo.LBoundSigned[LInfoIndex]), LoopInfo.IndVar[LInfoIndex]);

    BasicBlock *CondBB = Entry->getParent()->splitBasicBlock(Entry);
    CondBB->setName("for.cond");

    // The new entry is the start of the loop
    LoopEntryI = &CondBB->getUniquePredecessor()->front();

    IRB.SetInsertPoint(Entry);

    Value *IndVarVal = IRB.CreateLoad(
        IndVarTy,
        LoopInfo.IndVar[LInfoIndex]);
    Value *LoopCmp = nullptr;
    LoopCmp = buildCmpSignDependent(
      LoopInfo.LoopType[LInfoIndex], IRB, IndVarVal,
      LoopInfo.UBound[LInfoIndex].Result, LoopInfo.IndVarSigned[LInfoIndex], LoopInfo.UBoundSigned[LInfoIndex]).first;

    BasicBlock *BodyBB = Entry->getParent()->splitBasicBlock(Entry);
    BasicBlock *CollapseBB = BodyBB;
    for (size_t i = 0; i < LoopInfo.LBound.size(); ++i) {
      CollapseBB = CollapseBB->splitBasicBlock(Entry);
      CollapseIterBB.push_back(&CollapseBB->getUniquePredecessor()->front());
    }
    CollapseBB->setName("for.body");

    BasicBlock *EndBB = BasicBlock::Create(Ctx, "for.end", &F);
    IRB.SetInsertPoint(EndBB);
    IRB.CreateBr(Exit->getParent()->getUniqueSuccessor());

    // The new exit is the end of the loop
    LoopExitI = &EndBB->front();

    // Replace default br. by a conditional br. to task.body or task end
    Instruction *OldTerminator = CondBB->getTerminator();
    IRB.SetInsertPoint(OldTerminator);
    IRB.CreateCondBr(LoopCmp, BodyBB, EndBB);
    OldTerminator->eraseFromParent();

    BasicBlock *IncrBB = BasicBlock::Create(Ctx, "for.incr", &F);

    // Add a br. to for.cond
    IRB.SetInsertPoint(IncrBB);
    IndVarVal = IRB.CreateLoad(
        IndVarTy,
        LoopInfo.IndVar[LInfoIndex]);
    auto p = buildAddSignDependent(
      IRB, IndVarVal, LoopInfo.Step[LInfoIndex].Result, LoopInfo.IndVarSigned[LInfoIndex], LoopInfo.StepSigned[LInfoIndex]);
    IRB.CreateStore(createZSExtOrTrunc(IRB, p.first, IndVarTy, p.second), LoopInfo.IndVar[LInfoIndex]);
    IRB.CreateBr(CondBB);

    // Replace task end br. by a br. to for.incr
    OldTerminator = Exit->getParent()->getTerminator();
    assert(OldTerminator->getNumSuccessors() == 1);
    OldTerminator->setSuccessor(0, IncrBB);
  }

  Instruction *buildLoopForTask(
      const DirectiveEnvironment &DirEnv, Instruction *Entry,
      Instruction *Exit, const DirectiveLoopInfo &LoopInfo) {
    Instruction *LoopEntryI = nullptr;
    Instruction *LoopExitI = nullptr;

    SmallVector<Instruction *> CollapseIterBB;
    for (int i = LoopInfo.LBound.size() - 1; i >= 0; --i) {
      IRBuilder<> IRB(Entry);
      Type *IndVarTy = DirEnv.getDSAType(DirEnv.LoopInfo.IndVar[i]);
      LoopInfo.LBound[i].Result = IRB.CreateCall(LoopInfo.LBound[i].Fun, LoopInfo.LBound[i].Args);
      LoopInfo.UBound[i].Result = IRB.CreateCall(LoopInfo.UBound[i].Fun, LoopInfo.UBound[i].Args);
      LoopInfo.Step[i].Result = IRB.CreateCall(LoopInfo.Step[i].Fun, LoopInfo.Step[i].Args);

      buildLoopForTaskImpl(IndVarTy, Entry, Exit, LoopInfo, LoopEntryI, LoopExitI, CollapseIterBB, i);
      Entry = LoopEntryI;
      Exit = LoopExitI;
    }
    return LoopEntryI;
  }

  // Build a loop containing Single Entry Single Exit region Entry/Exit
  // and returns the WhileEntry and WhileExit
  void buildWhileForTaskImpl(
      Instruction *Entry, Instruction *Exit, const DirectiveWhileInfo &WhileInfo,
      Instruction *&WhileEntryI, Instruction *&WhileExitI) {
    BasicBlock *CondBB = Entry->getParent()->splitBasicBlock(Entry);
    CondBB->setName("while.cond");

    // The new entry is the start of the loop
    WhileEntryI = &CondBB->getUniquePredecessor()->front();

    IRBuilder<> IRB(Entry);

    WhileInfo.Result = IRB.CreateCall(WhileInfo.Fun, WhileInfo.Args);

    BasicBlock *BodyBB = Entry->getParent()->splitBasicBlock(Entry);
    BodyBB->setName("while.body");

    BasicBlock *EndBB = BasicBlock::Create(Ctx, "while.end", &F);
    IRB.SetInsertPoint(EndBB);
    IRB.CreateBr(Exit->getParent()->getUniqueSuccessor());

    // The new exit is the end of the loop
    WhileExitI = &EndBB->front();

    // Replace default br. by a conditional br. to task.body or task end
    Instruction *OldTerminator = CondBB->getTerminator();
    IRB.SetInsertPoint(OldTerminator);
    IRB.CreateCondBr(WhileInfo.Result, BodyBB, EndBB);
    OldTerminator->eraseFromParent();

    // Replace task end br. by a br. to while.cond
    OldTerminator = Exit->getParent()->getTerminator();
    assert(OldTerminator->getNumSuccessors() == 1);
    OldTerminator->setSuccessor(0, CondBB);
  }

  Instruction *buildWhileForTask(
      Instruction *Entry,
      Instruction *Exit, const DirectiveWhileInfo &WhileInfo) {
    Instruction *WhileEntryI = nullptr;
    Instruction *WhileExitI = nullptr;

    buildWhileForTaskImpl(Entry, Exit, WhileInfo, WhileEntryI, WhileExitI);
    return WhileEntryI;
  }

  void buildLoopForMultiDep(
      Instruction *Entry, Instruction *Exit,
      const MultiDependInfo *MultiDepInfo) {
    DenseMap<Value *, Value *> IndToRemap;
    for (size_t i = 0; i < MultiDepInfo->Iters.size(); i++) {
      Value *IndVar = MultiDepInfo->Iters[i];
      buildLoopForMultiDepImpl(
        F, Entry, Exit, IndVar, i, MultiDepInfo->ComputeMultiDepFun,
        MultiDepInfo->Args, IndToRemap);
    }
  }

  void buildLoopForMultiDepImpl(
      Function &Func,
      Instruction *Entry, Instruction *Exit,
      Value *IndVar, int IterSelector, Function *ComputeMultiDepFun,
      ArrayRef<Value *> Args, DenseMap<Value *, Value *> &IndToRemap) {

    DenseSet<Instruction *> InstrToRemap;

    SmallVector<Value *, 4> TmpArgs(Args.begin(), Args.end());
    // Select what iterator info we want to be computed
    TmpArgs.push_back(ConstantInt::get(Int64Ty, IterSelector));

    IRBuilder<> IRB(Entry);
    Type *IndVarTy = DirEnv.getDSAType(IndVar);

    AllocaInst *IndVarRemap = IRB.CreateAlloca(IndVarTy, nullptr, IndVar->getName() + ".remap");
    PostMoveInstructions.push_back(IndVarRemap);
    // Set the iterator to a valid value to avoid indexing discrete
    // assray with random bytes
    IRB.CreateStore(ConstantInt::get(IndVarTy, 0), IndVar);

    Instruction *ComputeMultiDepCall = IRB.CreateCall(ComputeMultiDepFun, TmpArgs);
    InstrToRemap.insert(ComputeMultiDepCall);
    Value *LBound = IRB.CreateExtractValue(ComputeMultiDepCall, IterSelector*(3 + 1) + 0);
    Value *UBound = IRB.CreateExtractValue(ComputeMultiDepCall, IterSelector*(3 + 1) + 2);
    Value *Incr = IRB.CreateExtractValue(ComputeMultiDepCall, IterSelector*(3 + 1) + 3);

    IRB.CreateStore(LBound, IndVar);

    BasicBlock *CondBB = IRB.saveIP().getBlock()->splitBasicBlock(IRB.saveIP().getPoint());
    CondBB->setName("for.cond");

    IRB.SetInsertPoint(Entry);

    Value *IndVarVal =
        IRB.CreateLoad(IndVarTy, IndVar);
    Value *LoopCmp = IRB.CreateICmpSLE(IndVarVal, UBound);

    BasicBlock *BodyBB = IRB.saveIP().getBlock()->splitBasicBlock(IRB.saveIP().getPoint());
    BodyBB->setName("for.body");

    // Build the remap value before runtime call and replace uses
    // NOTE: this is doable because when building multideps there's only a BasicBlock containing
    // the call to RT
    for (Instruction &I : *BodyBB) {
      I.replaceUsesOfWith(IndVar, IndVarRemap);
    }

    IRB.SetInsertPoint(&BodyBB->front());

    // Remap has to be computed every iteration
    {
      Instruction *ComputeMultiDepCall = IRB.CreateCall(ComputeMultiDepFun, TmpArgs);
      Value *Remap =
        IRB.CreateExtractValue(ComputeMultiDepCall, IterSelector*(3 + 1) + 1);
      InstrToRemap.insert(ComputeMultiDepCall);
      IRB.CreateStore(Remap, IndVarRemap);
    }

    // Replace default br. by a conditional br. to task.body or task end
    Instruction *OldTerminator = CondBB->getTerminator();
    IRB.SetInsertPoint(OldTerminator);
    IRB.CreateCondBr(LoopCmp, BodyBB, Exit->getParent()->getUniqueSuccessor());
    OldTerminator->eraseFromParent();

    BasicBlock *IncrBB = BasicBlock::Create(Ctx, "for.incr", &Func);

    // Add a br. to for.cond
    IRB.SetInsertPoint(IncrBB);

    IndVarVal =
        IRB.CreateLoad(IndVarTy, IndVar);
    Value *IndVarValPlusIncr = IRB.CreateAdd(IndVarVal, Incr);
    IRB.CreateStore(IndVarValPlusIncr, IndVar);
    IRB.CreateBr(CondBB);

    // Replace task end br. by a br. to for.incr
    OldTerminator = Exit->getParent()->getTerminator();
    assert(OldTerminator->getNumSuccessors() == 1);
    OldTerminator->setSuccessor(0, IncrBB);

    for (Instruction *I : InstrToRemap) {
      for (const auto &p : IndToRemap) {
        I->replaceUsesOfWith(p.first, p.second);
      }
    }

    // Record remapped iterator for future inner loop
    // replacing
    IndToRemap[IndVar] = IndVarRemap;
  }

  // Insert a new nanos6 task info registration in
  // the constructor (global ctor inserted) function
  void registerTaskInfo(Value *TaskInfoVar) {
    Function *Func = cast<Function>(nanos6Api::taskInfoRegisterCtorFuncCallee(M).getCallee());
    if (Func->empty()) {
      Func->setLinkage(GlobalValue::InternalLinkage);
      BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", Func);
      Instruction *RetInst = ReturnInst::Create(Ctx);
      RetInst->insertInto(EntryBB, EntryBB->end());

      appendToGlobalCtors(M, Func, 65535);
    }

    BasicBlock &Entry = Func->getEntryBlock();

    IRBuilder<> BBBuilder(&Entry.back());
    BBBuilder.CreateCall(nanos6Api::taskInfoRegisterFuncCallee(M), TaskInfoVar);
  }

  void unpackDestroyArgsAndRewrite(
      Function *UnpackFunc, const MapVector<Value *, size_t> &StructToIdxMap) {

    BasicBlock::Create(Ctx, "entry", UnpackFunc);
    BasicBlock &Entry = UnpackFunc->getEntryBlock();
    IRBuilder<> IRB(&Entry);

    for (const auto &DeinitMap : NonPODsInfo.Deinits) {
      auto *V = DeinitMap.first;
      Type *Ty = DirInfo.DirEnv.getDSAType(V);
      // Compute num elements
      Value *NSize = ConstantInt::get(Int64Ty, 1);
      if (isa<ArrayType>(Ty)) {
        while (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
          // Constant array
          Value *NumElems = ConstantInt::get(Int64Ty, ArrTy->getNumElements());
          NSize = IRB.CreateNUWMul(NSize, NumElems);
          Ty = ArrTy->getElementType();
        }
      } else if (VLADimsInfo.count(V)) {
        for (Value *Dim : VLADimsInfo.lookup(V))
          NSize = IRB.CreateNUWMul(NSize, UnpackFunc->getArg(StructToIdxMap.lookup(Dim)));
      }

      Value *FArg = UnpackFunc->getArg(StructToIdxMap.lookup(V));

      llvm::Function *Func = cast<Function>(DeinitMap.second);
      IRB.CreateCall(Func, ArrayRef<Value*>{FArg, NSize});
    }
    IRB.CreateRetVoid();
  }

  void unpackDepCallToRTImpl(
      const DependInfo *DepInfo,
      Function *UnpackFunc,
      ArrayRef<Value *> NewIndVarLBound, ArrayRef<Value *> NewIndVarUBound, bool IsTaskLoop,
      CallInst *& CallComputeDepStart, CallInst *& CallComputeDepEnd,
      CallInst *& RegisterDepCall) {

    BasicBlock &Entry = UnpackFunc->getEntryBlock();
    Instruction &RetI = Entry.back();
    IRBuilder<> BBBuilder(&RetI);

    Function *ComputeDepFun = cast<Function>(DepInfo->ComputeDepFun);
    CallComputeDepStart = BBBuilder.CreateCall(ComputeDepFun, DepInfo->Args);
    CallComputeDepEnd = BBBuilder.CreateCall(ComputeDepFun, DepInfo->Args);
    if (IsTaskLoop) {
      assert(!NewIndVarLBound.empty() && !NewIndVarUBound.empty() &&
        "Expected new lb/up in taskloop dependency");
      for (size_t i = 0; i < LoopInfo.LBound.size(); ++i) {
        CallComputeDepStart->replaceUsesOfWith(LoopInfo.IndVar[i], NewIndVarLBound[i]);
        CallComputeDepEnd->replaceUsesOfWith(LoopInfo.IndVar[i], NewIndVarUBound[i]);
      }
    }
    StructType *ComputeDepTy = cast<StructType>(ComputeDepFun->getReturnType());

    assert(ComputeDepTy->getNumElements() > 1 &&
      "Expected dependency base with dim_{size, start, end}");
    size_t NumDims = (ComputeDepTy->getNumElements() - 1)/3;

    llvm::Value *Base = BBBuilder.CreateExtractValue(CallComputeDepStart, 0);

    bool IsReduction = DepInfo->isReduction();

    SmallVector<Value *, 4> TaskDepAPICall;
    if (IsReduction) {
      TaskDepAPICall.push_back(DepInfo->RedKind);
      TaskDepAPICall.push_back(ConstantInt::get(Int32Ty, ReductionsInitCombInfo.lookup(Base).ReductionIndex));
    }

    Constant *RegionTextGV = BBBuilder.CreateGlobalStringPtr(DepInfo->RegionText);

    Value *Handler = &*(UnpackFunc->arg_end() - 1);
    TaskDepAPICall.push_back(Handler);
    TaskDepAPICall.push_back(ConstantInt::get(Int32Ty, DepInfo->SymbolIndex));
    TaskDepAPICall.push_back(RegionTextGV);
    TaskDepAPICall.push_back(Base);
    for (size_t i = 1; i < ComputeDepTy->getNumElements(); ) {
      // dimsize
      TaskDepAPICall.push_back(BBBuilder.CreateExtractValue(CallComputeDepStart, i++));
      // dimstart
      TaskDepAPICall.push_back(BBBuilder.CreateExtractValue(CallComputeDepStart, i++));
      // dimend
      TaskDepAPICall.push_back(BBBuilder.CreateExtractValue(CallComputeDepEnd, i++));
    }

    RegisterDepCall =
      BBBuilder.CreateCall(MultidepFactory.getMultidepFuncCallee(M, DepInfo->DepType, NumDims, IsReduction), TaskDepAPICall);

  }

  void unpackDepCallToRT(
      const DependInfo *DepInfo,
      Function *UnpackFunc, ArrayRef<Value *> NewIndVarLBound, ArrayRef<Value *> NewIndVarUBound,
      bool IsTaskLoop) {

    CallInst *CallComputeDepStart;
    CallInst *CallComputeDepEnd;
    CallInst *RegisterDepCall;
    unpackDepCallToRTImpl(
      DepInfo, UnpackFunc, NewIndVarLBound, NewIndVarUBound, IsTaskLoop,
      CallComputeDepStart, CallComputeDepEnd, RegisterDepCall);
  }

  void unpackMultiRangeCall(
      const MultiDependInfo *MultiDepInfo, Function *UnpackFunc,
      ArrayRef<Value *> NewIndVarLBound, ArrayRef<Value *> NewIndVarUBound,
      bool IsTaskLoop) {

    CallInst *CallComputeDepStart;
    CallInst *CallComputeDepEnd;
    CallInst *RegisterDepCall;
    unpackDepCallToRTImpl(
      MultiDepInfo, UnpackFunc, NewIndVarLBound, NewIndVarUBound, IsTaskLoop,
      CallComputeDepStart, CallComputeDepEnd, RegisterDepCall);

    // Build a BasicBlock conatining the compute_dep call and the dep. registration
    CallComputeDepStart->getParent()->splitBasicBlock(CallComputeDepStart);
    Instruction *AfterRegisterDepCall = RegisterDepCall->getNextNode();
    AfterRegisterDepCall->getParent()->splitBasicBlock(AfterRegisterDepCall);

    auto Args = MultiDepInfo->Args;

    // Build multidep loop based on loop iterators lower bound
    if (IsTaskLoop) {
      for (size_t i = 0; i < LoopInfo.IndVar.size(); i++)
        std::replace(Args.begin(), Args.end(), LoopInfo.IndVar[i], NewIndVarLBound[i]);
    }

    DenseMap<Value *, Value *> IndToRemap;
    for (size_t i = 0; i < MultiDepInfo->Iters.size(); i++) {
      Value *IndVar = MultiDepInfo->Iters[i];
      buildLoopForMultiDepImpl(
        *UnpackFunc, CallComputeDepStart, RegisterDepCall, IndVar, i, MultiDepInfo->ComputeMultiDepFun,
        Args, IndToRemap);
    }
  }

  void unpackDepsCallToRT(
      Function *UnpackFunc, bool IsTaskLoop, ArrayRef<Value *> NewIndVarLBound,
      ArrayRef<Value *> NewIndVarUBound) {

    for (auto &DepInfo : DependsInfo.List) {
      if (auto *MultiDepInfo = dyn_cast<MultiDependInfo>(DepInfo.get())) {
        // Multideps using loop iterator are assumed to be discrete
        if (IsTaskLoop && multidepUsesLoopIter(*MultiDepInfo)) {
          unpackMultiRangeCall(
            MultiDepInfo, UnpackFunc,
            NewIndVarLBound, NewIndVarLBound, IsTaskLoop);
        } else {
          unpackMultiRangeCall(
            MultiDepInfo, UnpackFunc,
            NewIndVarLBound, NewIndVarUBound, IsTaskLoop);
        }
      } else {
        unpackDepCallToRT(
          DepInfo.get(), UnpackFunc,
          NewIndVarLBound, NewIndVarUBound, IsTaskLoop);
      }
    }
  }

  Value *recoverTaskloopIterator(
      IRBuilder<> &IRB, ArrayRef<Value *> NormalizedUBs, int i, Value *Val, Type *Ty, Value *Storage) {
    // NOTE: NormalizedUBs values are nanos6_register_loop upper_bound type
    Value *Niters = ConstantInt::get(nanos6Api::Nanos6LoopBounds::getInstance(M).getUBType(), 1);
    for (size_t j = i + 1; j < LoopInfo.LBound.size(); ++j)
      Niters = IRB.CreateMul(Niters, NormalizedUBs[j]);

    // TMP = (LB|UB)/ProductUBs(i+1..n)
    auto pTmp = buildDivSignDependent(
      IRB, Val, Niters, LoopInfo.IndVarSigned[i], /*RHSSigned=*/false);

    // TMP * Step
    auto p = buildMulSignDependent(
      IRB, pTmp.first, LoopInfo.Step[i].Result, pTmp.second, LoopInfo.StepSigned[i]);

    // (TMP * Step) + OrigLBound
    auto pBodyIndVar = buildAddSignDependent(
      IRB, p.first, LoopInfo.LBound[i].Result, p.second, LoopInfo.LBoundSigned[i]);

    // TMP*ProductUBs(i+1..n)
    p = buildMulSignDependent(
      IRB, pTmp.first, Niters, pTmp.second, /*RHSSigned=*/false);

    // LoopInfo - TMP*ProductUBs(i+1..n)
    auto pLoopIndVar = buildSubSignDependent(
      IRB, Val, p.first, LoopInfo.IndVarSigned[i], p.second);

    // (LB|UB) = LoopInfo - TMP*ProductUBs(i+1..n)
    Val = pLoopIndVar.first;
    // BodyIndVar(i) = (TMP * Step) + OrigLBound
    IRB.CreateStore(createZSExtOrTrunc(IRB, pBodyIndVar.first, Ty, pBodyIndVar.second), Storage);
    return Val;
  }

  void unpackDepsAndRewrite(
      Function *UnpackFunc, const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(Ctx, "entry", UnpackFunc);
    BasicBlock &Entry = UnpackFunc->getEntryBlock();

    // add the terminator so IRBuilder inserts just before it
    Instruction *RetInst = ReturnInst::Create(Ctx);
    RetInst->insertInto(&Entry, Entry.end());

    SmallVector<Value *, 2> NewIndVarLBounds;
    SmallVector<Value *, 2> NewIndVarUBounds;

    bool IsTaskLoop = DirEnv.isOmpSsTaskLoopDirective();

    if (IsTaskLoop) {
      Type *IndVarTy = DirInfo.DirEnv.getDSAType(LoopInfo.IndVar[0]);

      IRBuilder<> IRB(&Entry.front());
      Value *LoopBounds = &*(UnpackFunc->arg_end() - 2);

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, 0);
      Value *LBoundField = IRB.CreateGEP(
          nanos6Api::Nanos6LoopBounds::getInstance(M).getType(),
          LoopBounds, Idx, "lb_gep");
      LBoundField = IRB.CreateLoad(
          nanos6Api::Nanos6LoopBounds::getInstance(M).getLBType(), LBoundField);
      LBoundField = IRB.CreateZExtOrTrunc(LBoundField, IndVarTy, "lb");

      Idx[1] = ConstantInt::get(Int32Ty, 1);
      Value *UBoundField = IRB.CreateGEP(
          nanos6Api::Nanos6LoopBounds::getInstance(M).getType(),
          LoopBounds, Idx, "ub_gep");
      UBoundField = IRB.CreateLoad(
          nanos6Api::Nanos6LoopBounds::getInstance(M).getUBType(), UBoundField);
      UBoundField = IRB.CreateZExtOrTrunc(UBoundField, IndVarTy);
      UBoundField = IRB.CreateSub(UBoundField, ConstantInt::get(IndVarTy, 1), "ub");

      SmallVector<Value *> NormalizedUBs(LoopInfo.UBound.size());
      // This is used for nanos6_create_loop
      // NOTE: all values have nanos6 upper_bound type
      ComputeLoopBounds(&Entry.back(), NormalizedUBs);

      for (size_t i = 0; i < LoopInfo.LBound.size(); ++i) {
        Value *NewIndVarLBound = IRB.CreateAlloca(IndVarTy, nullptr, LoopInfo.IndVar[i]->getName() + ".lb");
        Value *NewIndVarUBound = IRB.CreateAlloca(IndVarTy, nullptr, LoopInfo.IndVar[i]->getName() + ".ub");

        LBoundField = recoverTaskloopIterator(IRB, NormalizedUBs, i, LBoundField, IndVarTy, NewIndVarLBound);
        UBoundField = recoverTaskloopIterator(IRB, NormalizedUBs, i, UBoundField, IndVarTy, NewIndVarUBound);

        NewIndVarLBounds.push_back(NewIndVarLBound);
        NewIndVarUBounds.push_back(NewIndVarUBound);
      }
    }

    // Insert RT call before replacing uses
    unpackDepsCallToRT(UnpackFunc, IsTaskLoop, NewIndVarLBounds, NewIndVarUBounds);

    for (BasicBlock &BB : *UnpackFunc) {
      for (Instruction &I : BB) {
        Function::arg_iterator AI = UnpackFunc->arg_begin();
        for (auto It = StructToIdxMap.begin();
               It != StructToIdxMap.end(); ++It, ++AI) {
          if (isReplaceableValue(It->first))
            I.replaceUsesOfWith(It->first, &*AI);
        }
      }
    }
  }

  void unpackCostAndRewrite(
      Function *UnpackFunc, const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(Ctx, "entry", UnpackFunc);
    BasicBlock &Entry = UnpackFunc->getEntryBlock();
    Instruction *RetInst = ReturnInst::Create(Ctx);
    RetInst->insertInto(&Entry, Entry.end());
    IRBuilder<> BBBuilder(&UnpackFunc->getEntryBlock().back());
    Value *Constraints = &*(UnpackFunc->arg_end() - 1);
    Value *Idx[2];
    Idx[0] = Constant::getNullValue(Int32Ty);
    Idx[1] = Constant::getNullValue(Int32Ty);

    Value *GEPConstraints =
        BBBuilder.CreateGEP(nanos6Api::Nanos6TaskConstraints::getInstance(M).getType(),
                            Constraints, Idx, "gep_" + Constraints->getName());
    Value *Cost = BBBuilder.CreateCall(CostInfo.Fun, CostInfo.Args);
    Value *CostCast = BBBuilder.CreateZExt(Cost, nanos6Api::Nanos6TaskConstraints::getInstance(M).getCostType());
    BBBuilder.CreateStore(CostCast, GEPConstraints);
    for (Instruction &I : Entry) {
      Function::arg_iterator AI = UnpackFunc->arg_begin();
      for (auto It = StructToIdxMap.begin();
             It != StructToIdxMap.end(); ++It, ++AI) {
        if (isReplaceableValue(It->first))
          I.replaceUsesOfWith(It->first, &*AI);
      }
    }
  }

  void unpackPriorityAndRewrite(
      Function *UnpackFunc, const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(Ctx, "entry", UnpackFunc);
    BasicBlock &Entry = UnpackFunc->getEntryBlock();
    Instruction *RetInst = ReturnInst::Create(Ctx);
    RetInst->insertInto(&Entry, Entry.end());
    IRBuilder<> BBBuilder(&UnpackFunc->getEntryBlock().back());
    Value *PriorityArg = &*(UnpackFunc->arg_end() - 1);

    Value *Priority = BBBuilder.CreateCall(PriorityInfo.Fun, PriorityInfo.Args);
    Value *PrioritySExt = BBBuilder.CreateSExt(Priority, Int64Ty);
    BBBuilder.CreateStore(PrioritySExt, PriorityArg);
    for (Instruction &I : Entry) {
      Function::arg_iterator AI = UnpackFunc->arg_begin();
      for (auto It = StructToIdxMap.begin();
             It != StructToIdxMap.end(); ++It, ++AI) {
        if (isReplaceableValue(It->first))
          I.replaceUsesOfWith(It->first, &*AI);
      }
    }
  }

  void unpackOnreadyAndRewrite(
      Function *UnpackFunc, const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(Ctx, "entry", UnpackFunc);
    BasicBlock &Entry = UnpackFunc->getEntryBlock();
    Instruction *RetInst = ReturnInst::Create(Ctx);
    RetInst->insertInto(&Entry, Entry.end());
    IRBuilder<> BBBuilder(&UnpackFunc->getEntryBlock().back());

    BBBuilder.CreateCall(OnreadyInfo.Fun, OnreadyInfo.Args);
    for (Instruction &I : Entry) {
      Function::arg_iterator AI = UnpackFunc->arg_begin();
      for (auto It = StructToIdxMap.begin();
             It != StructToIdxMap.end(); ++It, ++AI) {
        if (isReplaceableValue(It->first))
          I.replaceUsesOfWith(It->first, &*AI);
      }
    }
  }

  void unpackWhileAndRewrite(
      Function *UnpackFunc, const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(Ctx, "entry", UnpackFunc);
    BasicBlock &Entry = UnpackFunc->getEntryBlock();
    Instruction *RetInst = ReturnInst::Create(Ctx);
    RetInst->insertInto(&Entry, Entry.end());
    IRBuilder<> BBBuilder(&UnpackFunc->getEntryBlock().back());
    Value *ResArg = &*(UnpackFunc->arg_end() - 1);

    Value *Res = BBBuilder.CreateCall(WhileInfo.Fun, WhileInfo.Args);
    Value *ResSExt = BBBuilder.CreateZExt(Res, Int8Ty);
    BBBuilder.CreateStore(ResSExt, ResArg);

    for (Instruction &I : Entry) {
      Function::arg_iterator AI = UnpackFunc->arg_begin();
      for (auto It = StructToIdxMap.begin();
             It != StructToIdxMap.end(); ++It, ++AI) {
        if (isReplaceableValue(It->first))
          I.replaceUsesOfWith(It->first, &*AI);
      }
    }
  }

  void unpackReleaseDepCallToRT(
      const DependInfo *DepInfo, Instruction *InsertPt) {

    IRBuilder<> BBBuilder(InsertPt);

    Function *ComputeDepFun = cast<Function>(DepInfo->ComputeDepFun);
    Value *CallComputeDep = BBBuilder.CreateCall(ComputeDepFun, DepInfo->Args);
    StructType *ComputeDepTy = cast<StructType>(ComputeDepFun->getReturnType());

    assert(ComputeDepTy->getNumElements() > 1 && "Expected dependency base with dim_{size, start, end}");
    size_t NumDims = (ComputeDepTy->getNumElements() - 1)/3;

    llvm::Value *Base = BBBuilder.CreateExtractValue(CallComputeDep, 0);

    SmallVector<Value *, 4> TaskDepAPICall;
    TaskDepAPICall.push_back(Base);
    for (size_t i = 1; i < ComputeDepTy->getNumElements(); ) {
      // dimsize
      TaskDepAPICall.push_back(BBBuilder.CreateExtractValue(CallComputeDep, i++));
      // dimstart
      TaskDepAPICall.push_back(BBBuilder.CreateExtractValue(CallComputeDep, i++));
      // dimend
      TaskDepAPICall.push_back(BBBuilder.CreateExtractValue(CallComputeDep, i++));
    }

    BBBuilder.CreateCall(
      MultidepFactory.getReleaseMultidepFuncCallee(M, DepInfo->DepType, NumDims), TaskDepAPICall);

  }

  void unpackReleaseDepsCallToRT() {
    for (auto &DepInfo : DependsInfo.List) {
      assert(!isa<MultiDependInfo>(DepInfo.get()) && "release directive does not support multideps");
      unpackReleaseDepCallToRT(DepInfo.get(), DirInfo.Entry);
    }
  }

  // TypeList[i] <-> NameList[i]
  // ExtraTypeList[i] <-> ExtraNameList[i]
  Function *createUnpackOlFunction(std::string Name,
                                 ArrayRef<Type *> TypeList,
                                 ArrayRef<StringRef> NameList,
                                 ArrayRef<Type *> ExtraTypeList,
                                 ArrayRef<StringRef> ExtraNameList,
                                 bool IsTask = false) {
    Type *RetTy = Type::getVoidTy(Ctx);

    SmallVector<Type *, 4> AggTypeList;
    AggTypeList.append(TypeList.begin(), TypeList.end());
    AggTypeList.append(ExtraTypeList.begin(), ExtraTypeList.end());

    SmallVector<StringRef, 4> AggNameList;
    AggNameList.append(NameList.begin(), NameList.end());
    AggNameList.append(ExtraNameList.begin(), ExtraNameList.end());

    FunctionType *FuncType =
      FunctionType::get(RetTy, AggTypeList, /*IsVarArgs=*/ false);

    Function *FuncVar = Function::Create(
        FuncType, GlobalValue::InternalLinkage, F.getAddressSpace(),
        Name, &M);

    // Build debug info in task unpack
    if (IsTask) {
      DICompileUnit *CU = nullptr;
      DIFile *File = nullptr;
      DISubprogram *OldSP = F.getSubprogram();
      if (OldSP) {
        CU = OldSP->getUnit();
        File = OldSP->getFile();
      }
      DIBuilder DIB(M, /*AllowUnresolved=*/false, CU);
      auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray(std::nullopt));
      DISubprogram::DISPFlags SPFlags = DISubprogram::SPFlagDefinition |
                                        DISubprogram::SPFlagOptimized |
                                        DISubprogram::SPFlagLocalToUnit;
      DISubprogram *NewSP = DIB.createFunction(
          CU, FuncVar->getName(), FuncVar->getName(), File,
          /*LineNo=*/0, SPType, /*ScopeLine=*/0, DINode::FlagZero, SPFlags);
      FuncVar->setSubprogram(NewSP);
      DIB.finalizeSubprogram(NewSP);
    }

    // Set names for arguments.
    Function::arg_iterator AI = FuncVar->arg_begin();
    for (unsigned i = 0, e = AggNameList.size(); i != e; ++i, ++AI)
      AI->setName(AggNameList[i]);

    return FuncVar;
  }

  // Rewrites task_args using address_translation
  void translateDep(
      IRBuilder<> &IRBTranslate, IRBuilder<> &IRBReload, Value *DSA,
      bool IsShared, Value *&UnpackedDSA, Type *UnpackedDSATy,
      Value *AddrTranslationTable, int SymbolIndex) {

    // nanos6_address_translation_entry_t *address_translation_table
    Type *TaskAddrTranslationEntryTy = nanos6Api::Nanos6TaskAddrTranslationEntry::getInstance(M).getType();

    llvm::Value *DepBase = UnpackedDSA;
    llvm::Value *AddrToTranslate = DepBase;
    if (!IsShared) {
      assert(!isa<LoadInst>(UnpackedDSA));
      DepBase = IRBTranslate.CreateLoad(UnpackedDSATy, DepBase);
      AddrToTranslate = DepBase;
      // HUGE FIXME! (1)
      // flang POINTER used in dependencies are firstprivate
      // fir.box. We want to translate the buffer, so here
      // we are assuming the first field is the buffer
      // In the future we way do this in a generic way
      if (UnpackedDSATy->isStructTy())
        AddrToTranslate = IRBTranslate.CreateExtractValue(DepBase, 0, "agg.ptr");
    }

    Value *Idx[2];
    Idx[0] = ConstantInt::get(Int32Ty, SymbolIndex);
    Idx[1] = Constant::getNullValue(Int32Ty);
    Value *LocalAddr = IRBTranslate.CreateGEP(
        TaskAddrTranslationEntryTy,
        AddrTranslationTable, Idx, "local_lookup_" + DSA->getName());
    LocalAddr = IRBTranslate.CreateLoad(Int64Ty, LocalAddr);

    Idx[1] = ConstantInt::get(Int32Ty, 1);
    Value *DeviceAddr = IRBTranslate.CreateGEP(
        TaskAddrTranslationEntryTy,
        AddrTranslationTable, Idx, "device_lookup_" + DSA->getName());
    DeviceAddr = IRBTranslate.CreateLoad(Int64Ty, DeviceAddr);

    // Res = device_addr + (DSA_addr - local_addr)
    Value *Translation =
        IRBTranslate.CreateGEP(Int8Ty, AddrToTranslate, IRBTranslate.CreateNeg(LocalAddr));
    Translation = IRBTranslate.CreateGEP(Int8Ty, Translation, DeviceAddr);

    // Store the translation
    if (IsShared) {
      auto *LUnpackedDSA = cast<LoadInst>(UnpackedDSA);
      IRBTranslate.CreateStore(Translation, LUnpackedDSA->getPointerOperand());
      // Reload what we have translated
      UnpackedDSA = IRBReload.CreateLoad(PtrTy, LUnpackedDSA->getPointerOperand());
    } else {
      // HUGE FIXME!
      // Same note as (1)
      if (UnpackedDSATy->isStructTy())
        Translation = IRBTranslate.CreateInsertValue(DepBase, Translation, 0);
      IRBTranslate.CreateStore(Translation, UnpackedDSA);
    }
  }

  // Given a Outline Function assuming that task args are the first parameter, and
  // DSAInfo and VLADimsInfo, it unpacks task args in Outline and fills UnpackedList
  // and UnpackedToTypeMap with those Values and Types, used to call Unpack Functions
  void unpackDSAsWithVLADims(
      Type *TaskArgsTy, Function *OlFunc, const MapVector<Value *, size_t> &StructToIdxMap,
      SmallVectorImpl<Value *> &UnpackedList, MapVector<Value *, Type *> &UnpackedToTypeMap) {
    UnpackedList.clear();
    UnpackedList.resize(StructToIdxMap.size());

    IRBuilder<> BBBuilder(&OlFunc->getEntryBlock());
    Function::arg_iterator AI = OlFunc->arg_begin();
    Value *OlDepsFuncTaskArgs = &*AI++;
    for (const auto &Pair : DSAInfo.Shared) {
      Value *V = Pair.first;

      size_t IdxI = StructToIdxMap.lookup(V);

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, IdxI);
      Value *GEP = BBBuilder.CreateGEP(
          TaskArgsTy,
          OlDepsFuncTaskArgs, Idx, "gep_" + V->getName());
      Value *LGEP =
          BBBuilder.CreateLoad(PtrTy, GEP,
                               "load_" + GEP->getName());

      UnpackedList[HasDevice ? IdxI - DeviceArgsSize : IdxI] = LGEP;
    }
    for (const auto &Pair : DSAInfo.Private) {
      Value *V = Pair.first;
      Type *Ty = Pair.second;

      size_t IdxI = StructToIdxMap.lookup(V);

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, IdxI);
      Value *GEP = BBBuilder.CreateGEP(
          TaskArgsTy,
          OlDepsFuncTaskArgs, Idx, "gep_" + V->getName());

      // VLAs
      if (VLADimsInfo.count(V))
        GEP = BBBuilder.CreateLoad(PtrTy, GEP, "load_" + GEP->getName());

      UnpackedList[HasDevice ? IdxI - DeviceArgsSize : IdxI] = GEP;
      UnpackedToTypeMap[GEP] = Ty;
    }
    for (const auto &Pair : DSAInfo.Firstprivate) {
      Value *V = Pair.first;
      Type *Ty = Pair.second;

      size_t IdxI = StructToIdxMap.lookup(V);

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, IdxI);
      Value *GEP = BBBuilder.CreateGEP(
          TaskArgsTy,
          OlDepsFuncTaskArgs, Idx, "gep_" + V->getName());

      // VLAs
      if (VLADimsInfo.count(V))
        GEP = BBBuilder.CreateLoad(PtrTy, GEP, "load_" + GEP->getName());

      UnpackedList[HasDevice ? IdxI - DeviceArgsSize : IdxI] = GEP;
      UnpackedToTypeMap[GEP] = Ty;
    }
    for (Value *V : CapturedInfo) {
      size_t IdxI = StructToIdxMap.lookup(V);

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, IdxI);
      Value *GEP = BBBuilder.CreateGEP(
          TaskArgsTy,
          OlDepsFuncTaskArgs, Idx, "capt_gep" + V->getName());
      Value *LGEP =
          BBBuilder.CreateLoad(V->getType(), GEP, "load_" + GEP->getName());
      UnpackedList[HasDevice ? IdxI - DeviceArgsSize : IdxI] = LGEP;
    }
  }

  // task_args cannot be modified. This function creates
  // new variables where the translation is performed.
  void dupTranlationNeededArgs(
      IRBuilder<> &IRBEntry,
      const std::map<Value *, int> &DepSymToIdx,
      const MapVector<Value *, size_t> &StructToIdxMap,
      SmallVector<Value *, 4> &UnpackParams,
      const MapVector<Value *, Type *> &UnpackParamsToTypesMap) {

    for (const auto &p : DepSymToIdx) {
      Value *DepBaseDSA = p.first;
      size_t Idx = StructToIdxMap.lookup(DepBaseDSA);
      if (HasDevice)
        Idx -= DeviceArgsSize;
      Value *UnpackedDSA = UnpackParams[Idx];
      if (auto *LUnpackedDSA = dyn_cast<LoadInst>(UnpackedDSA)) {
        Value *NewDepBaseDSA =
          IRBEntry.CreateAlloca(
            PtrTy, nullptr, "tlate." + LUnpackedDSA->getName());
        IRBEntry.CreateStore(LUnpackedDSA, NewDepBaseDSA);

        UnpackParams[Idx] = IRBEntry.CreateLoad(PtrTy, NewDepBaseDSA);
      } else {
        Value *NewDepBaseDSA =
          IRBEntry.CreateAlloca(
            UnpackParamsToTypesMap.lookup(UnpackedDSA), nullptr, "tlate." + UnpackedDSA->getName());
        IRBEntry.CreateStore(
          IRBEntry.CreateLoad(
            UnpackParamsToTypesMap.lookup(UnpackedDSA), UnpackedDSA),
          NewDepBaseDSA);

        UnpackParams[Idx] = NewDepBaseDSA;
      }
    }
  }

  // Given an Outline and Unpack Functions it unpacks DSAs in Outline
  // and builds a call to Unpack
  void olCallToUnpack(
      Type *TaskArgsTy, const MapVector<Value *, size_t> &StructToIdxMap,
      Function *OlFunc, Function *UnpackFunc, bool IsTaskFunc=false) {
    BasicBlock::Create(Ctx, "entry", OlFunc);

    // First arg is the nanos_task_args
    Function::arg_iterator AI = OlFunc->arg_begin();
    AI++;
    SmallVector<Value *, 4> UnpackParams;
    MapVector<Value *, Type *> UnpackParamsToTypesMap;
    unpackDSAsWithVLADims(TaskArgsTy, OlFunc, StructToIdxMap, UnpackParams, UnpackParamsToTypesMap);
    while (AI != OlFunc->arg_end()) {
      UnpackParams.push_back(&*AI++);
    }

    if (IsTaskFunc) {
      BasicBlock *IfThenBB = BasicBlock::Create(Ctx, "", OlFunc);
      BasicBlock *IfEndBB = BasicBlock::Create(Ctx, "", OlFunc);

      // Builders
      IRBuilder<> IRBEntry(&OlFunc->getEntryBlock());
      IRBuilder<> IRBIfThen(IfThenBB);
      IRBuilder<> IRBIfEnd(IfEndBB);

      // Build the skeleton
      Value *AddrTranslationTable = &*(OlFunc->arg_end() - 1);
      Value *Cmp = IRBEntry.CreateICmpNE(
        AddrTranslationTable, Constant::getNullValue(AddrTranslationTable->getType()));
      IRBEntry.CreateCondBr(Cmp, IfThenBB, IfEndBB);

      IRBIfThen.CreateBr(IfEndBB);
      IRBIfEnd.CreateRetVoid();

      // Reset insert points.
      IRBEntry.SetInsertPoint(cast<Instruction>(Cmp));
      IRBIfThen.SetInsertPoint(IfThenBB->getTerminator());
      IRBIfEnd.SetInsertPoint(IfEndBB->getTerminator());

      // Preserve the params to still take profit of UnpackParamsToTypesMap
      // NOTE: this assumes UnpackParams can be indexed with StructToIdxMap
      SmallVector<Value *, 4> UnpackParamsCopy(UnpackParams);

      dupTranlationNeededArgs(
          IRBEntry, DirInfo.DirEnv.DepSymToIdx, StructToIdxMap, UnpackParams, UnpackParamsToTypesMap);

      for (const auto &p : DirInfo.DirEnv.DepSymToIdx) {
        Value *DepBaseDSA = p.first;
        int SymbolIndex = p.second;

        bool IsShared = DirInfo.DirEnv.DSAInfo.Shared.count(DepBaseDSA);

        size_t Idx = StructToIdxMap.lookup(DepBaseDSA);
        if (HasDevice)
          Idx -= DeviceArgsSize;

        translateDep(
          IRBIfThen, IRBIfEnd, DepBaseDSA,
          IsShared, UnpackParams[Idx], UnpackParamsToTypesMap.lookup(UnpackParamsCopy[Idx]),
          AddrTranslationTable, SymbolIndex);
      }
      // Build TaskUnpackCall with the translated values
      IRBIfEnd.CreateCall(UnpackFunc, UnpackParams);
    } else {
      // Build TaskUnpackCall
      IRBuilder<> IRBEntry(&OlFunc->getEntryBlock());
      IRBEntry.CreateCall(UnpackFunc, UnpackParams);
      IRBEntry.CreateRetVoid();
    }
  }

  // Copy task_args from src to dst, calling copyctors or ctors if
  // nonpods
  void duplicateArgs(
      const MapVector<Value *, size_t> &StructToIdxMap, Function *OlFunc, StructType *TaskArgsTy) {
    BasicBlock::Create(Ctx, "entry", OlFunc);
    IRBuilder<> IRB(&OlFunc->getEntryBlock());

    Function::arg_iterator AI = OlFunc->arg_begin();
    Value *TaskArgsSrc = &*AI++;
    Value *TaskArgsDst = &*AI++;
    Value *TaskArgsDstL = IRB.CreateLoad(PtrTy, TaskArgsDst);

    SmallVector<VLAAlign, 2> VLAAlignsInfo;
    computeVLAsAlignOrder(VLAAlignsInfo);

    Value *TaskArgsStructSizeOf = ConstantInt::get(Int64Ty, DL.getTypeAllocSize(TaskArgsTy));

    // TODO: this forces an alignment of 16 for VLAs
    {
      const int ALIGN = 16;
      TaskArgsStructSizeOf =
        IRB.CreateNUWAdd(TaskArgsStructSizeOf,
                         ConstantInt::get(Int64Ty, ALIGN - 1));
      TaskArgsStructSizeOf =
        IRB.CreateAnd(TaskArgsStructSizeOf,
                      IRB.CreateNot(ConstantInt::get(Int64Ty, ALIGN - 1)));
    }

    Value *TaskArgsDstLi8IdxGEP =
      IRB.CreateGEP(Int8Ty, TaskArgsDstL, TaskArgsStructSizeOf, "args_end");

    // First point VLAs to its according space in task args
    for (const VLAAlign& VAlign : VLAAlignsInfo) {
      auto *V = VAlign.V;
      Type *Ty = DirEnv.getDSAType(V);
      Align TyAlign = VAlign.TyAlign;

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, StructToIdxMap.lookup(V));
      Value *GEP =
          IRB.CreateGEP(TaskArgsTy,
                        TaskArgsDstL, Idx, "gep_dst_" + V->getName());

      // Point VLA in task args to an aligned position of the extra space allocated
      IRB.CreateAlignedStore(TaskArgsDstLi8IdxGEP, GEP, TyAlign);
      // Skip current VLA size
      unsigned SizeB = DL.getTypeAllocSize(Ty);
      Value *VLASize = ConstantInt::get(Int64Ty, SizeB);
      for (auto *Dim : VLADimsInfo.lookup(V)) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(Int32Ty);
        Idx[1] = ConstantInt::get(Int32Ty, StructToIdxMap.lookup(Dim));
        Value *GEPDst =
            IRB.CreateGEP(TaskArgsTy,
                          TaskArgsDstL, Idx, "gep_dst_" + Dim->getName());
        GEPDst = IRB.CreateLoad(Int64Ty, GEPDst);
        VLASize = IRB.CreateNUWMul(VLASize, GEPDst);
      }
      TaskArgsDstLi8IdxGEP = IRB.CreateGEP(
        Int8Ty, TaskArgsDstLi8IdxGEP, VLASize);
    }

    for (const auto &Pair : DSAInfo.Shared) {
      Value *V = Pair.first;

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, StructToIdxMap.lookup(V));
      Value *GEPSrc = IRB.CreateGEP(
          TaskArgsTy,
          TaskArgsSrc, Idx, "gep_src_" + V->getName());
      Value *GEPDst = IRB.CreateGEP(
          TaskArgsTy,
          TaskArgsDstL, Idx, "gep_dst_" + V->getName());
      IRB.CreateStore(
          IRB.CreateLoad(PtrTy, GEPSrc),
          GEPDst);
    }
    for (const auto &Pair : DSAInfo.Private) {
      Value *V = Pair.first;
      Type *Ty = Pair.second;
      // Call custom constructor generated in clang in non-pods
      // Leave pods unititialized
      auto It = NonPODsInfo.Inits.find(V);
      if (It != NonPODsInfo.Inits.end()) {
        // Compute num elements
        Value *NSize = ConstantInt::get(Int64Ty, 1);
        if (isa<ArrayType>(Ty)) {
          while (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
            // Constant array
            Value *NumElems = ConstantInt::get(Int64Ty, ArrTy->getNumElements());
            NSize = IRB.CreateNUWMul(NSize, NumElems);
            Ty = ArrTy->getElementType();
          }
        } else if (VLADimsInfo.count(V)) {
          for (auto *Dim : VLADimsInfo.lookup(V)) {
            Value *Idx[2];
            Idx[0] = Constant::getNullValue(Int32Ty);
            Idx[1] = ConstantInt::get(Int32Ty, StructToIdxMap.lookup(Dim));
            Value *GEPSrc =
                IRB.CreateGEP(TaskArgsTy,
                              TaskArgsSrc, Idx, "gep_src_" + Dim->getName());
            GEPSrc = IRB.CreateLoad(Int64Ty, GEPSrc);
            NSize = IRB.CreateNUWMul(NSize, GEPSrc);
          }
        }

        Value *Idx[2];
        Idx[0] = Constant::getNullValue(Int32Ty);
        Idx[1] = ConstantInt::get(Int32Ty, StructToIdxMap.lookup(V));
        Value *GEP =
            IRB.CreateGEP(TaskArgsTy,
                          TaskArgsDstL, Idx, "gep_" + V->getName());

        // VLAs
        if (VLADimsInfo.count(V))
          GEP = IRB.CreateLoad(PtrTy, GEP);

        IRB.CreateCall(FunctionCallee(cast<Function>(It->second)), ArrayRef<Value*>{GEP, NSize});
      }
    }
    for (const auto &Pair : DSAInfo.Firstprivate) {
      Value *V = Pair.first;
      Type *Ty = Pair.second;
      Align TyAlign = DL.getPrefTypeAlign(Ty);

      // Compute num elements
      Value *NSize = ConstantInt::get(Int64Ty, 1);
      if (isa<ArrayType>(Ty)) {
        while (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
          // Constant array
          Value *NumElems = ConstantInt::get(Int64Ty, ArrTy->getNumElements());
          NSize = IRB.CreateNUWMul(NSize, NumElems);
          Ty = ArrTy->getElementType();
        }
      } else if (VLADimsInfo.count(V)) {
        for (auto *Dim : VLADimsInfo.lookup(V)) {
          Value *Idx[2];
          Idx[0] = Constant::getNullValue(Int32Ty);
          Idx[1] = ConstantInt::get(Int32Ty, StructToIdxMap.lookup(Dim));
          Value *GEPSrc =
              IRB.CreateGEP(TaskArgsTy,
                            TaskArgsSrc, Idx, "gep_src_" + Dim->getName());
          // VLA sizes are always i64
          GEPSrc = IRB.CreateLoad(Int64Ty, GEPSrc);
          NSize = IRB.CreateNUWMul(NSize, GEPSrc);
        }
      }

      // call custom copy constructor generated in clang in non-pods
      // do a memcpy if pod
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, StructToIdxMap.lookup(V));
      Value *GEPSrc =
          IRB.CreateGEP(TaskArgsTy,
                        TaskArgsSrc, Idx, "gep_src_" + V->getName());
      Value *GEPDst =
          IRB.CreateGEP(TaskArgsTy,
                        TaskArgsDstL, Idx, "gep_dst_" + V->getName());

      // VLAs
      if (VLADimsInfo.count(V)) {
        GEPSrc =
            IRB.CreateLoad(PtrTy, GEPSrc);
        GEPDst =
            IRB.CreateLoad(PtrTy, GEPDst);
      }

      auto It = NonPODsInfo.Copies.find(V);
      if (It != NonPODsInfo.Copies.end()) {
        // Non-POD
        llvm::Function *Func = cast<Function>(It->second);
        IRB.CreateCall(Func, ArrayRef<Value*>{/*Src=*/GEPSrc, /*Dst=*/GEPDst, NSize});
      } else {
        unsigned SizeB = DL.getTypeAllocSize(Ty);
        Value *NSizeB = IRB.CreateNUWMul(NSize, ConstantInt::get(Int64Ty, SizeB));
        IRB.CreateMemCpy(GEPDst, TyAlign, GEPSrc, TyAlign, NSizeB);
      }
    }
    for (Value *V : CapturedInfo) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, StructToIdxMap.lookup(V));
      Value *GEPSrc =
          IRB.CreateGEP(TaskArgsTy,
                        TaskArgsSrc, Idx, "capt_gep_src_" + V->getName());
      Value *GEPDst =
          IRB.CreateGEP(TaskArgsTy,
                        TaskArgsDstL, Idx, "capt_gep_dst_" + V->getName());
      IRB.CreateStore(
          IRB.CreateLoad(V->getType(), GEPSrc),
          GEPDst);
    }

    IRB.CreateRetVoid();
  }

  Value *computeTaskArgsVLAsExtraSizeOf(IRBuilder<> &IRB) {
    Value *Sum = ConstantInt::get(Int64Ty, 0);
    for (const auto &VLAWithDimsMap : VLADimsInfo) {
      // Skip shareds because they don't need space in task_args
      if (DSAInfo.Shared.count(VLAWithDimsMap.first))
        continue;
      Type *Ty = DirEnv.getDSAType(VLAWithDimsMap.first);
      unsigned SizeB = DL.getTypeAllocSize(Ty);
      Value *ArraySize = ConstantInt::get(Int64Ty, SizeB);
      for (auto *V : VLAWithDimsMap.second) {
        ArraySize = IRB.CreateNUWMul(ArraySize, V);
      }
      Sum = IRB.CreateNUWAdd(Sum, ArraySize);
    }
    return Sum;
  }

  StructType *createTaskArgsType(
      MapVector<Value *, size_t> &StructToIdxMap, StringRef Str) {
    SmallVector<Type *, 4> TaskArgsMemberTy;
    size_t TaskArgsIdx = 0;

    SmallPtrSet<Value *, 4> Visited;
    if (!DeviceInfo.empty()) {
      // Add device info

      // size_t global_size0;
      // ...
      // size_t global_sizeN-1;

      // size_t local_size0;
      // ...
      // size_t local_sizeN-1;
      const size_t FullNdrangeLength = 6;
      for (size_t i = 0; i < FullNdrangeLength; ++i) {
        TaskArgsMemberTy.push_back(Int64Ty);
        TaskArgsIdx++;
      }
      // size_t shm_size;
      TaskArgsMemberTy.push_back(Int64Ty);
      TaskArgsIdx++;

      // Give priority to the device all arguments
      for (Value *V : DeviceInfo.CallOrder) {
        StructToIdxMap[V] = TaskArgsIdx++;
        Visited.insert(V);
      }
    }

    // Extend TaskArgsMemberTy including the CallOrder values.
    TaskArgsMemberTy.resize(TaskArgsMemberTy.size() + Visited.size());

    // Private and Firstprivate must be stored in the struct
    // Captured values (i.e. VLA dimensions) are not pointers
    for (const auto &Pair : DSAInfo.Shared) {
      Value *V = Pair.first;
      if (!Visited.count(V)) {
        TaskArgsMemberTy.push_back(PtrTy);
        StructToIdxMap[V] = TaskArgsIdx++;
      } else {
        TaskArgsMemberTy[StructToIdxMap.lookup(V)] = PtrTy;
      }
    }
    for (const auto &Pair : DSAInfo.Private) {
      Value *V = Pair.first;
      Type *Ty = Pair.second;
      if (!Visited.count(V)) {
        // VLAs
        if (VLADimsInfo.count(V))
          TaskArgsMemberTy.push_back(PtrTy);
        else
          TaskArgsMemberTy.push_back(Ty);
        StructToIdxMap[V] = TaskArgsIdx++;
      } else {
        // VLAs
        if (VLADimsInfo.count(V))
          TaskArgsMemberTy[StructToIdxMap.lookup(V)] = PtrTy;
        else
          TaskArgsMemberTy[StructToIdxMap.lookup(V)] = Ty;
      }
    }
    for (const auto &Pair : DSAInfo.Firstprivate) {
      Value *V = Pair.first;
      Type *Ty = Pair.second;
      if (!Visited.count(V)) {
        // VLAs
        if (VLADimsInfo.count(V))
          TaskArgsMemberTy.push_back(PtrTy);
        else
          TaskArgsMemberTy.push_back(Ty);
        StructToIdxMap[V] = TaskArgsIdx++;
      } else {
        // VLAs
        if (VLADimsInfo.count(V))
          TaskArgsMemberTy[StructToIdxMap.lookup(V)] = PtrTy;
        else
          TaskArgsMemberTy[StructToIdxMap.lookup(V)] = Ty;
      }
    }
    for (Value *V : CapturedInfo) {
      assert(!V->getType()->isPointerTy() && "Captures are not pointers");
      if (!Visited.count(V)) {
        TaskArgsMemberTy.push_back(V->getType());
        StructToIdxMap[V] = TaskArgsIdx++;
      } else {
        TaskArgsMemberTy[StructToIdxMap.lookup(V)] = V->getType();
      }
    }
    return StructType::create(Ctx, TaskArgsMemberTy, Str);
  }

  // Useful to get lists of info needed for creating functions, assign
  // names to the values and so on.
  // Also computes the list of sizeofs and offsets for the members
  // needed for devices.
  void getTaskArgsInfo(
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy,
      SmallVector<Type *, 4> &TaskTypeList, SmallVector<StringRef, 4> &TaskNameList,
      GlobalVariable *&SizeofTableVar, GlobalVariable *&OffsetTableVar,
      GlobalVariable *&ArgIdxTableVar) {

    for (auto It = TaskArgsToStructIdxMap.begin();
           It != TaskArgsToStructIdxMap.end(); ++It) {
      Value *V = It->first;
      TaskTypeList.push_back(V->getType());
      TaskNameList.push_back(V->getName());
    }

    // int *sizeof_table;
    // Type *SizeofTableTy = nanos6Api::Nanos6TaskInfo::getInstance(M).getSizeofTableDataType();
    SmallVector<Constant *, 4> TaskSizeofList;
    ArrayRef<Type *> TaskElementTypes = TaskArgsTy->elements();
    // Remove the device elements since we do not want them
    // in the sizeof info
    if (!DeviceInfo.empty())
      TaskElementTypes = TaskElementTypes.drop_front(DeviceArgsSize);
    for (Type *Ty : TaskElementTypes) {
      TaskSizeofList.push_back(
        ConstantInt::get(
          Int32Ty,
          DL.getTypeStoreSize(Ty).getFixedValue()));
    }
    SizeofTableVar =
      new GlobalVariable(M, ArrayType::get(Int32Ty, TaskTypeList.size()),
        /*isConstant=*/true, GlobalVariable::InternalLinkage,
        ConstantArray::get(ArrayType::get(Int32Ty, TaskTypeList.size()),
          TaskSizeofList),
        ("sizeof_table_var_" + F.getName()).str());
    SizeofTableVar->setAlignment(Align(64));

    // int *offset_table;
    // Type *OffsetTableTy = nanos6Api::Nanos6TaskInfo::getInstance(M).getOffsetTableDataType();
    ArrayRef<TypeSize> MemberOffsetsList =
      DL.getStructLayout(TaskArgsTy)->getMemberOffsets();
    if (!DeviceInfo.empty())
      MemberOffsetsList = MemberOffsetsList.drop_front(DeviceArgsSize);
    SmallVector<Constant *, 4> TaskOffsetList;
    for (const TypeSize &val : MemberOffsetsList) {
      TaskOffsetList.push_back(
        ConstantInt::get(Int32Ty, val));
    }
    OffsetTableVar =
      new GlobalVariable(M, ArrayType::get(Int32Ty, TaskTypeList.size()),
        /*isConstant=*/true, GlobalVariable::InternalLinkage,
        ConstantArray::get(ArrayType::get(Int32Ty, TaskTypeList.size()),
          TaskOffsetList),
        ("offset_table_var_" + F.getName()).str());

    // int *arg_idx_table;
    // Type *ArgIdxTableTy = nanos6Api::Nanos6TaskInfo::getInstance(M).getArgIdxTableDataType();
    SmallVector<Constant *, 4> TaskArgIdxList(DirEnv.DependsInfo.NumSymbols);
    for (const auto &p : DirEnv.DepSymToIdx) {
      Value *DepBase = p.first;
      int SymbolIndex = p.second;
      int Idx = TaskArgsToStructIdxMap.lookup(DepBase);
      if (!DeviceInfo.empty())
        Idx -= DeviceArgsSize;
      TaskArgIdxList[SymbolIndex] =
        ConstantInt::get(Int32Ty, Idx);
    }
    ArgIdxTableVar =
      new GlobalVariable(M, ArrayType::get(Int32Ty, DirEnv.DependsInfo.NumSymbols),
        /*isConstant=*/true, GlobalVariable::InternalLinkage,
        ConstantArray::get(ArrayType::get(Int32Ty, DirEnv.DependsInfo.NumSymbols),
          TaskArgIdxList),
        ("arg_idx_table_var_" + F.getName()).str());
  }

  struct VLAAlign {
    Value *V;
    Align TyAlign;
  };

  // Greater alignemt go first
  void computeVLAsAlignOrder(SmallVectorImpl<VLAAlign> &VLAAlignsInfo) {
    for (const auto &VLAWithDimsMap : VLADimsInfo) {
      // Skip shareds because they don't need space in task_args
      if (DSAInfo.Shared.count(VLAWithDimsMap.first))
        continue;
      auto *V = VLAWithDimsMap.first;
      Type *Ty = DirEnv.getDSAType(V);

      Align TyAlign = DL.getPrefTypeAlign(Ty);

      auto It = VLAAlignsInfo.begin();
      while (It != VLAAlignsInfo.end() && It->TyAlign >= TyAlign)
        ++It;

      VLAAlignsInfo.insert(It, {V, TyAlign});
    }
  }

  void lowerTaskwait() {
    // 1. Create Taskwait function Type
    IRBuilder<> IRB(DirInfo.Entry);
    FunctionCallee Func = M.getOrInsertFunction(
        "nanos6_taskwait", IRB.getVoidTy(), PtrTy);
    // 2. Build String
    unsigned Line = 0;
    unsigned Col = 0;
    DebugLoc DLoc = DirInfo.Entry->getDebugLoc();
    if (DLoc) {
      Line = DLoc.getLine();
      Col = DLoc.getCol();
    }

    std::string FileNamePlusLoc = (M.getSourceFileName()
                                   + ":" + Twine(Line)
                                   + ":" + Twine(Col)).str();
    Constant *Nanos6TaskwaitLocStr = IRB.CreateGlobalStringPtr(FileNamePlusLoc);

    // 3. Insert the call
    IRB.CreateCall(Func, {Nanos6TaskwaitLocStr});
    // 4. Remove the intrinsic
    DirInfo.Entry->eraseFromParent();
  }

  void lowerRelease() {
    unpackReleaseDepsCallToRT();
    // Remove the intrinsic
    DirInfo.Entry->eraseFromParent();
  }

  Value *mapValue(Value *V, ValueToValueMapTy &VMap) {
    // Remap the value if necessary.
    ValueToValueMapTy::iterator I = VMap.find(V);
    if (I != VMap.end())
      return I->second;
    return V;
  }

  void rewriteDirInfoForFinal(
      DirectiveLoopInfo &LoopInfo, ValueToValueMapTy &VMap) {
    for (size_t i = 0; i < LoopInfo.LBound.size(); ++i) {
      LoopInfo.IndVar[i] = mapValue(LoopInfo.IndVar[i], VMap);
      for (size_t j = 0; j < LoopInfo.LBound[i].Args.size(); ++j)
        LoopInfo.LBound[i].Args[j] = mapValue(LoopInfo.LBound[i].Args[j], VMap);
      for (size_t j = 0; j < LoopInfo.UBound[i].Args.size(); ++j)
        LoopInfo.UBound[i].Args[j] = mapValue(LoopInfo.UBound[i].Args[j], VMap);
      for (size_t j = 0; j < LoopInfo.Step[i].Args.size(); ++j)
        LoopInfo.Step[i].Args[j] = mapValue(LoopInfo.Step[i].Args[j], VMap);
    }
  }

  // Final codes need to float nested directives constant allocas
  // Loop directives need to float body constant allocas
  // This function annotates them assuming all the allocas are placed just
  // after the Entry intrinsic (because clang does that).
  void gatherConstAllocas(Instruction *Entry) {
    BasicBlock::iterator It = Entry->getParent()->begin();
    // Skip Entry intrinsic
    ++It;
    // Allocas inside the task are put just after Entry
    // intrinsic
    while (It != Entry->getParent()->end()) {
      if (auto *II = dyn_cast<AllocaInst>(&*It)) {
        if (isa<ConstantInt>(II->getArraySize()))
          PostMoveInstructions.push_back(II);
      } else {
        break;
      }
      ++It;
    }
  }

  void rewriteDirInfoForFinal(
      DirectiveWhileInfo &WhileInfo, ValueToValueMapTy &VMap) {
    for (size_t i = 0; i < WhileInfo.Args.size(); ++i)
      WhileInfo.Args[i] = mapValue(WhileInfo.Args[i], VMap);
  }

  Function *createDestroyArgsOlFunc(
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, ArrayRef<Type *> TaskTypeList,
      ArrayRef<StringRef> TaskNameList) {

    // Do not create anything
    if (NonPODsInfo.Deinits.empty())
      return nullptr;

    Function *UnpackDestroyArgsFuncVar
      = createUnpackOlFunction(
        ("nanos6_unpacked_destroy_" + F.getName()).str(),
        TaskTypeList, TaskNameList, {}, {});
    unpackDestroyArgsAndRewrite(UnpackDestroyArgsFuncVar, TaskArgsToStructIdxMap);

    // nanos6_unpacked_destroy_* END

    // nanos6_ol_destroy_* START

    Function *OlDestroyArgsFuncVar
      = createUnpackOlFunction(
        ("nanos6_ol_destroy_" + F.getName()).str(),
        {PtrTy}, {"task_args"}, {}, {});
    olCallToUnpack(TaskArgsTy, TaskArgsToStructIdxMap, OlDestroyArgsFuncVar, UnpackDestroyArgsFuncVar);

    return OlDestroyArgsFuncVar;

  }

  Function *createDuplicateArgsOlFunc(
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, ArrayRef<Type *> TaskTypeList,
      ArrayRef<StringRef> TaskNameList) {

    // Do not create anything if all are PODs
    // and there are no VLAs.
    // The runtime will perform a memcpy
    if (NonPODsInfo.Inits.empty() &&
        NonPODsInfo.Copies.empty() &&
        VLADimsInfo.empty())
      return nullptr;

    Function *OlDuplicateArgsFuncVar
      = createUnpackOlFunction(("nanos6_ol_duplicate_" + F.getName()).str(),
                               {PtrTy, PtrTy}, {"task_args_src", "task_args_dst"},
                               {}, {});
    duplicateArgs(TaskArgsToStructIdxMap, OlDuplicateArgsFuncVar, TaskArgsTy);

    return OlDuplicateArgsFuncVar;
  }

  void createTaskFuncOl(
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy,
      ArrayRef<Type *> TaskTypeList, ArrayRef<StringRef> TaskNameList,
      bool IsLoop, Function *&OlFunc, Function *&UnpackFunc) {

    SmallVector<Type *, 4> TaskExtraTypeList;
    SmallVector<StringRef, 4> TaskExtraNameList;

    if (IsLoop) {
      TaskExtraTypeList.push_back(PtrTy);
      TaskExtraNameList.push_back("loop_bounds");
    } else {
      // void *device_env
      TaskExtraTypeList.push_back(PtrTy);
      TaskExtraNameList.push_back("device_env");
    }
    TaskExtraTypeList.push_back(PtrTy);
    TaskExtraNameList.push_back("address_translation_table");

    // CodeExtractor will create a entry block for us
    UnpackFunc = createUnpackOlFunction(
      ("nanos6_unpacked_task_region_" + F.getName()).str(),
      TaskTypeList, TaskNameList, TaskExtraTypeList, TaskExtraNameList, /*IsTask=*/true);

    OlFunc = createUnpackOlFunction(
      ("nanos6_ol_task_region_" + F.getName()).str(),
      {PtrTy}, {"task_args"}, TaskExtraTypeList, TaskExtraNameList);

    olCallToUnpack(TaskArgsTy, TaskArgsToStructIdxMap, OlFunc, UnpackFunc, /*IsTaskFunc=*/true);
  }

  Function *createDepsOlFunc(
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, ArrayRef<Type *> TaskTypeList,
      ArrayRef<StringRef> TaskNameList) {

    // Do not do anything if there are no dependencies
    if (DependsInfo.List.empty())
      return nullptr;

    SmallVector<Type *, 4> TaskExtraTypeList;
    SmallVector<StringRef, 4> TaskExtraNameList;

    // nanos6_loop_bounds_t *const loop_bounds
    TaskExtraTypeList.push_back(PtrTy);
    TaskExtraNameList.push_back("loop_bounds");
    // void *handler
    TaskExtraTypeList.push_back(PtrTy);
    TaskExtraNameList.push_back("handler");


    Function *UnpackDepsFuncVar
      = createUnpackOlFunction(("nanos6_unpacked_deps_" + F.getName()).str(),
                               TaskTypeList, TaskNameList,
                               TaskExtraTypeList, TaskExtraNameList);
    unpackDepsAndRewrite(UnpackDepsFuncVar, TaskArgsToStructIdxMap);

    Function *OlDepsFuncVar
      = createUnpackOlFunction(("nanos6_ol_deps_" + F.getName()).str(),
                               {PtrTy}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);
    olCallToUnpack(TaskArgsTy, TaskArgsToStructIdxMap, OlDepsFuncVar, UnpackDepsFuncVar);

    return OlDepsFuncVar;
  }

  Function *createConstraintsOlFunc(
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, ArrayRef<Type *> TaskTypeList,
      ArrayRef<StringRef> TaskNameList) {

    if (!CostInfo.Fun)
      return nullptr;

    SmallVector<Type *, 4> TaskExtraTypeList;
    SmallVector<StringRef, 4> TaskExtraNameList;

    // nanos6_task_constraints_t *constraints
    TaskExtraTypeList.push_back(PtrTy);
    TaskExtraNameList.push_back("constraints");

    Function *UnpackConstraintsFuncVar = createUnpackOlFunction(("nanos6_unpacked_constraints_" + F.getName()).str(),
                               TaskTypeList, TaskNameList,
                               TaskExtraTypeList, TaskExtraNameList);
    unpackCostAndRewrite(UnpackConstraintsFuncVar, TaskArgsToStructIdxMap);

    Function *OlConstraintsFuncVar
      = createUnpackOlFunction(("nanos6_ol_constraints_" + F.getName()).str(),
                               {PtrTy}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);
    olCallToUnpack(TaskArgsTy, TaskArgsToStructIdxMap, OlConstraintsFuncVar, UnpackConstraintsFuncVar);

    return OlConstraintsFuncVar;
  }

  // Checks if the LoopInfo[i] depends on other iterator
  // NOTE: this assumes compute_lb/ub/step have an iterator as
  // an arguments if and only if it is used.
  bool isLoopIteratorDepenent(unsigned i) {
    for (size_t j = 0; j < i; ++j) {
      Value *IndVar = LoopInfo.IndVar[j];
      for (Value *V : LoopInfo.LBound[i].Args)
        if (V == IndVar)
          return true;
      for (Value *V : LoopInfo.UBound[i].Args)
        if (V == IndVar)
          return true;
      for (Value *V : LoopInfo.Step[i].Args)
        if (V == IndVar)
          return true;
    }
    return false;
  }

  bool multidepUsesLoopIter(const MultiDependInfo& MultiDepInfo) {
    for (const auto *V : MultiDepInfo.Args) {
      for (size_t i = 0; i < LoopInfo.IndVar.size(); ++i)
        if (V == LoopInfo.IndVar[i])
          return true;
    }
    return false;
  }

  bool hasMultidepUsingLoopIter() {
    for (auto &DepInfo : DependsInfo.List) {
      if (const auto *MultiDepInfo = dyn_cast<MultiDependInfo>(DepInfo.get())) {
        if (multidepUsesLoopIter(*MultiDepInfo))
          return true;
      }
    }
    return false;
  }

  Function *createPriorityOlFunc(
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, ArrayRef<Type *> TaskTypeList,
      ArrayRef<StringRef> TaskNameList) {

    if (!PriorityInfo.Fun)
      return nullptr;

    SmallVector<Type *, 4> TaskExtraTypeList;
    SmallVector<StringRef, 4> TaskExtraNameList;

    // nanos6_priority_t *priority
    // long int *priority
    TaskExtraTypeList.push_back(PtrTy);
    TaskExtraNameList.push_back("priority");

    Function *UnpackPriorityFuncVar = createUnpackOlFunction(("nanos6_unpacked_priority_" + F.getName()).str(),
                               TaskTypeList, TaskNameList,
                               TaskExtraTypeList, TaskExtraNameList);
    unpackPriorityAndRewrite(UnpackPriorityFuncVar, TaskArgsToStructIdxMap);

    Function *OlPriorityFuncVar
      = createUnpackOlFunction(("nanos6_ol_priority_" + F.getName()).str(),
                               {PtrTy}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);
    olCallToUnpack(TaskArgsTy, TaskArgsToStructIdxMap, OlPriorityFuncVar, UnpackPriorityFuncVar);

    return OlPriorityFuncVar;
  }

  Function *createOnreadyOlFunc(
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, ArrayRef<Type *> TaskTypeList,
      ArrayRef<StringRef> TaskNameList) {

    if (!OnreadyInfo.Fun)
      return nullptr;

    Function *UnpackOnreadyFuncVar = createUnpackOlFunction(("nanos6_unpacked_onready_" + F.getName()).str(),
                               TaskTypeList, TaskNameList, {}, {});
    unpackOnreadyAndRewrite( UnpackOnreadyFuncVar, TaskArgsToStructIdxMap);

    Function *OlOnreadyFuncVar
      = createUnpackOlFunction(("nanos6_ol_onready_" + F.getName()).str(),
                               {PtrTy}, {"task_args"},
                               {}, {});
    olCallToUnpack(TaskArgsTy, TaskArgsToStructIdxMap, OlOnreadyFuncVar, UnpackOnreadyFuncVar);

    return OlOnreadyFuncVar;
  }

  Function *createTaskIterWhileCondOlFunc(
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, ArrayRef<Type *> TaskTypeList,
      ArrayRef<StringRef> TaskNameList) {

    if (WhileInfo.empty())
      return nullptr;

    SmallVector<Type *, 4> TaskExtraTypeList;
    SmallVector<StringRef, 4> TaskExtraNameList;

    // uint8_t *result
    TaskExtraTypeList.push_back(PtrTy);
    TaskExtraNameList.push_back("result");

    Function *UnpackWhileFuncVar = createUnpackOlFunction(("nanos6_unpacked_while_cond_" + F.getName()).str(),
                               TaskTypeList, TaskNameList,
                               TaskExtraTypeList, TaskExtraNameList);
    unpackWhileAndRewrite(UnpackWhileFuncVar, TaskArgsToStructIdxMap);

    Function *OlWhileFuncVar
      = createUnpackOlFunction(("nanos6_ol_while_cond_" + F.getName()).str(),
                               {PtrTy}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);
    olCallToUnpack(TaskArgsTy, TaskArgsToStructIdxMap, OlWhileFuncVar, UnpackWhileFuncVar);

    return OlWhileFuncVar;
  }

  // typedef enum {
  //         //! Specifies that the task will be a final task
  //         nanos6_final_task = (1 << 0),
  //         //! Specifies that the task is in "if(0)" mode → !If
  //         nanos6_if_0_task = (1 << 1),
  //         //! Specifies that the task is really a taskloop
  //         nanos6_taskloop_task = (1 << 2),
  //         //! Specifies that the task is really a taskfor
  //         nanos6_taskfor_task = (1 << 3),
  //         //! Specifies that the task is really a taskiter
  //         nanos6_taskiter_task = (1 << 4),
  //         //! Specifies that the task has the "wait" clause
  //         nanos6_waiting_task = (1 << 5),
  //         //! Specifies that the args_block is preallocated from user side
  //         nanos6_preallocated_args_block = (1 << 6),
  //         //! Specifies that the task has been verified by the user, hence it doesn't need runtime linting
  //         nanos6_verified_task = (1 << 7)
  //         //! Specifies that the task has the "update" clause
  //         nanos6_update_task = (1 << 8)
  // } nanos6_task_flag_t;
  Value *computeTaskFlags(IRBuilder<> &IRB) {
    Value *TaskFlagsVar = ConstantInt::get(Int64Ty, 0);
    if (DirEnv.Final) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateZExt(DirEnv.Final,
                         Int64Ty));
    }
    // Device Ndrange tasks default to if(0) in
    // final context
    if (InFinalCtx) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
            IRB.CreateZExt(
              IRB.getTrue(), Int64Ty),
              1));
    } else if (DirEnv.If) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
            IRB.CreateZExt(
              IRB.CreateICmpEQ(DirEnv.If, IRB.getFalse()),
              Int64Ty),
              1));
    }
    if (DirEnv.isOmpSsTaskLoopDirective()) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
              ConstantInt::get(Int64Ty, 1),
              2));
    }
    if (DirEnv.isOmpSsTaskForDirective()) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
              ConstantInt::get(Int64Ty, 1),
              3));
    }
    if (DirEnv.Wait) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
            IRB.CreateZExt(
              DirEnv.Wait,
              Int64Ty),
              4));
    }
    if (DirEnv.isOmpSsTaskIterDirective()) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
              ConstantInt::get(Int64Ty, 1),
              7));
    }
    if (LoopInfo.Update) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
            IRB.CreateZExt(
              LoopInfo.Update,
              Int64Ty),
              8));
    }
    return TaskFlagsVar;
  }

  static void computeBBsBetweenEntryExit(
      SetVector<Instruction *> &TaskBBs,
      Instruction *Entry, Instruction *Exit) {

    SmallVector<BasicBlock*, 8> Worklist;
    SmallPtrSet<BasicBlock*, 8> Visited;

    BasicBlock *ExitBB = Exit->getParent();

    // 2. Gather BB between entry and exit (is there any function/util to do this?)
    Worklist.push_back(Entry->getParent());
    Visited.insert(Entry->getParent());
    TaskBBs.insert(&Entry->getParent()->front());
    while (!Worklist.empty()) {
      auto WIt = Worklist.begin();
      BasicBlock *BB = *WIt;
      Worklist.erase(WIt);

      for (auto It = succ_begin(BB); It != succ_end(BB); ++It) {
        if (!Visited.count(*It) && *It != ExitBB) {
          Worklist.push_back(*It);
          Visited.insert(*It);
          TaskBBs.insert(&It->front());
        }
      }
    }
  }

  void buildUnpackedLoopForTask(
      DirectiveLoopInfo &NewLoopInfo,
      Instruction *& NewEntryI, Instruction *& NewExitI,
      SmallVectorImpl<Value *> &NormalizedUBs,
      SmallVectorImpl<Instruction *> &CollapseIterBB) {

    // This is used for nanos6_create_loop
    // NOTE: all values have nanos6 upper_bound type
    ComputeLoopBounds(DirInfo.Entry, NormalizedUBs);

    IRBuilder<> IRB(DirInfo.Entry);

    Type *IndVarTy = DirInfo.DirEnv.getDSAType(LoopInfo.IndVar[0]);
    // Non collapsed loops build a loop using the original type
    NewLoopInfo.LBoundSigned[0] = LoopInfo.IndVarSigned[0];
    NewLoopInfo.UBoundSigned[0] = LoopInfo.IndVarSigned[0];
    NewLoopInfo.StepSigned[0] = LoopInfo.IndVarSigned[0];
    if (LoopInfo.LBound.size() > 1) {
      // Collapsed loops build a loop using size_t type to avoid overflows
      IndVarTy = Int64Ty;
      NewLoopInfo.LBoundSigned[0] = 0;
      NewLoopInfo.UBoundSigned[0] = 0;
      NewLoopInfo.StepSigned[0] = 0;
    }
    // In reality we do not need this except for buildind the loop here.
    // These values will be replaced by nanos6 bounds
    NewLoopInfo.LBound[0].Result = createZSExtOrTrunc(IRB, LoopInfo.LBound[0].Result, IndVarTy, LoopInfo.LBoundSigned[0]);
    NewLoopInfo.UBound[0].Result = createZSExtOrTrunc(IRB, LoopInfo.UBound[0].Result, IndVarTy, LoopInfo.UBoundSigned[0]);
    // unpacked_task_region loops are always step 1
    NewLoopInfo.Step[0].Result = ConstantInt::get(IndVarTy, 1);

    NewLoopInfo.IndVar[0] = IRB.CreateAlloca(IndVarTy, nullptr, "loop");
    // unpacked_task_region loops are always SLT
    NewLoopInfo.LoopType[0] = DirectiveLoopInfo::LT;
    buildLoopForTaskImpl(IndVarTy, DirInfo.Entry, DirInfo.Exit, NewLoopInfo, NewEntryI, NewExitI, CollapseIterBB);
  }

  Function *rewriteUsesBrAndGetOmpSsUnpackFunc(
    const DirectiveLoopInfo &NewLoopInfo,
    SmallVectorImpl<Value *> &NormalizedUBs,
    SmallVectorImpl<Instruction *> &CollapseIterBB,
    Function *UnpackTaskFuncVar,
    const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
    Instruction *Exit,
    // Placeholders
    BasicBlock *header, BasicBlock *newRootNode, BasicBlock *newHeader,
    Function *oldFunction, const SetVector<BasicBlock *> &Blocks) {
    UnpackTaskFuncVar->insert(UnpackTaskFuncVar->end(), newRootNode);

    if (DirEnv.isOmpSsLoopDirective()) {
      Type *OrigIndVarTy = DirEnv.getDSAType(LoopInfo.IndVar[0]);
      Type *NewIndVarTy = OrigIndVarTy;
      // Collapsed loops build a loop using size_t type to avoid overflows
      if (LoopInfo.LBound.size() > 1)
        NewIndVarTy = Int64Ty;

      IRBuilder<> IRB(&header->front());
      Value *LoopBounds = &*(UnpackTaskFuncVar->arg_end() - 2);

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, 0);
      Value *LBoundField =
          IRB.CreateGEP(nanos6Api::Nanos6LoopBounds::getInstance(M).getType(),
                        LoopBounds, Idx, "lb_gep");
      LBoundField = IRB.CreateLoad(
          nanos6Api::Nanos6LoopBounds::getInstance(M).getLBType(), LBoundField);
      LBoundField = IRB.CreateZExtOrTrunc(LBoundField, NewIndVarTy, "lb");

      Idx[1] = ConstantInt::get(Int32Ty, 1);
      Value *UBoundField =
          IRB.CreateGEP(nanos6Api::Nanos6LoopBounds::getInstance(M).getType(),
                        LoopBounds, Idx, "ub_gep");
      UBoundField = IRB.CreateLoad(
          nanos6Api::Nanos6LoopBounds::getInstance(M).getUBType(), UBoundField);
      UBoundField = IRB.CreateZExtOrTrunc(UBoundField, NewIndVarTy, "ub");

      // Replace loop bounds of the indvar, loop cond. and loop incr.
      // NOTE: incr. does not need to be replaced because all loops have step 1.
      rewriteUsesInBlocksWithPred(
        NewLoopInfo.LBound[0].Result, LBoundField,
        [&Blocks](Instruction *I) { return Blocks.count(I->getParent()); });
      rewriteUsesInBlocksWithPred(
        NewLoopInfo.UBound[0].Result, UBoundField,
        [&Blocks](Instruction *I) { return Blocks.count(I->getParent()); });

      // Now we can set
      // TMP = LoopIndVar/ProductUBs(i+1..n)
      // LoopIndVar = LoopIndVar - TMP*PProductUBs(i+1..n)
      // BodyIndVar(i) = (TMP * Step) + OrigLBound
      Value *NormVal = nullptr;
      for (size_t i = 0; i < LoopInfo.LBound.size(); ++i) {
        Instruction *TmpEntry = CollapseIterBB[i];

        IRBuilder<> LoopBodyIRB(TmpEntry);
        if (!i)
          NormVal = LoopBodyIRB.CreateLoad(
              NewIndVarTy,
              NewLoopInfo.IndVar[0]);

        NormVal = recoverTaskloopIterator(LoopBodyIRB, NormalizedUBs, i, NormVal, OrigIndVarTy, LoopInfo.IndVar[i]);

        if (isLoopIteratorDepenent(i)) {
          // Create a check to skip unwanted iterations due to collapse guess
          Instruction *UBoundResult = LoopBodyIRB.CreateCall(LoopInfo.UBound[i].Fun, LoopInfo.UBound[i].Args);

          Instruction *IndVarVal = LoopBodyIRB.CreateLoad(
              DirEnv.getDSAType(LoopInfo.IndVar[i]),
              LoopInfo.IndVar[i]);
          Value *LoopCmp = nullptr;
          LoopCmp = buildCmpSignDependent(
            LoopInfo.LoopType[i], LoopBodyIRB, IndVarVal,
            UBoundResult, LoopInfo.IndVarSigned[i], LoopInfo.UBoundSigned[i]).first;

          // The IncrBB is the successor of BodyBB
          BasicBlock *BodyBB = Exit->getParent();
          Instruction *IncrBBI = &BodyBB->getUniqueSuccessor()->front();

          // Next iterator computation or BodyBB
          BasicBlock *NextBB = CollapseIterBB[i]->getParent()->getUniqueSuccessor();

          // Replace the branch
          Instruction *Terminator = TmpEntry->getParent()->getTerminator();
          LoopBodyIRB.SetInsertPoint(Terminator);
          LoopBodyIRB.CreateCondBr(LoopCmp, NextBB, IncrBBI->getParent());
          Terminator->eraseFromParent();
        }
      }
    }

    // Create an iterator to name all of the arguments we inserted.
    Function::arg_iterator AI = UnpackTaskFuncVar->arg_begin();
    // Rewrite all users of the TaskArgsToStructIdxMap in the extracted region to use the
    // arguments (or appropriate addressing into struct) instead.
    for (auto It = TaskArgsToStructIdxMap.begin();
           It != TaskArgsToStructIdxMap.end(); ++It) {
      Value *RewriteVal = &*AI++;
      Value *Val = It->first;

      if (auto *GV = dyn_cast<GlobalValue>(Val)) {
        // Convert all constant expr inside task body to instructions
        constantExprToInstruction(GV, Blocks);
      }

      if (isReplaceableValue(Val)) {
        rewriteUsesInBlocksWithPred(
          Val, RewriteVal,
          [&Blocks](Instruction *I) { return Blocks.count(I->getParent()); });
      }
    }

    // Rewrite branches from basic blocks outside of the task region to blocks
    // inside the region to use the new label (newHeader) since the task region
    // will be outlined
    rewriteUsesInBlocksWithPred(
      header, newHeader,
      [&Blocks, &oldFunction](Instruction *I) {
        return (I->isTerminator() && !Blocks.count(I->getParent()) &&
                I->getParent()->getParent() == oldFunction);
      });

    return UnpackTaskFuncVar;
  };

  CallInst *emitOmpSsCaptureAndSubmitTask(
      DebugLoc &DLoc, Type *TaskArgsTy,
      MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      Value *TaskInfoVar, Value *TaskInvInfoVar,
      // Placeholders
      Function *newFunction, BasicBlock *codeReplacer,
      const SetVector<BasicBlock *> &Blocks) {
    IRBuilder<> IRB(codeReplacer);
    // Set debug info from the task entry to all instructions
    IRB.SetCurrentDebugLocation(DLoc);

    // Add a branch to the next basic block after the task region
    // and replace the terminator that exits the task region
    // Since this is a single entry single exit region this should
    // be done once.
    BasicBlock *NewRetBB = nullptr;
    for (BasicBlock *Block : Blocks) {
      Instruction *DirInfo = Block->getTerminator();
      for (unsigned i = 0, e = DirInfo->getNumSuccessors(); i != e; ++i)
        if (!Blocks.count(DirInfo->getSuccessor(i))) {
          assert(!NewRetBB && "More than one exit in task code");

          BasicBlock *OldTarget = DirInfo->getSuccessor(i);
          // Create branch to next BB after the task region
          IRB.CreateBr(OldTarget);

          NewRetBB = BasicBlock::Create(Ctx, ".exitStub", newFunction);
          IRBuilder<> (NewRetBB).CreateRetVoid();

          // rewrite the original branch instruction with this new target
          DirInfo->setSuccessor(i, NewRetBB);
        }
    }

    // Here we have a valid codeReplacer BasicBlock with its terminator
    IRB.SetInsertPoint(codeReplacer->getTerminator());

    AllocaInst *TaskArgsVar = IRB.CreateAlloca(PtrTy);
    PostMoveInstructions.push_back(TaskArgsVar);
    Value *TaskFlagsVar = computeTaskFlags(IRB);
    AllocaInst *TaskPtrVar = IRB.CreateAlloca(PtrTy);
    PostMoveInstructions.push_back(TaskPtrVar);

    Value *TaskArgsStructSizeOf = IRB.getInt64(DL.getTypeAllocSize(TaskArgsTy));

    // TODO: this forces an alignment of 16 for VLAs
    {
      const int ALIGN = 16;
      TaskArgsStructSizeOf =
        IRB.CreateNUWAdd(TaskArgsStructSizeOf, IRB.getInt64(ALIGN - 1));
      TaskArgsStructSizeOf =
        IRB.CreateAnd(TaskArgsStructSizeOf, IRB.CreateNot(IRB.getInt64(ALIGN - 1)));
    }

    Value *TaskArgsVLAsExtraSizeOf = computeTaskArgsVLAsExtraSizeOf(IRB);
    Value *TaskArgsSizeOf = IRB.CreateNUWAdd(TaskArgsStructSizeOf, TaskArgsVLAsExtraSizeOf);

    Type *NumDependenciesTy = Int64Ty;
    Instruction *NumDependencies = IRB.CreateAlloca(NumDependenciesTy, nullptr, "num.deps");
    PostMoveInstructions.push_back(NumDependencies);

    if (DirEnv.isOmpSsTaskLoopDirective() && hasMultidepUsingLoopIter()) {
      // If taskloop has a multidep using the loop iterator
      // NumDeps = -1
      IRB.CreateStore(IRB.getInt64(-1), NumDependencies);
    } else {
      IRB.CreateStore(IRB.getInt64(0), NumDependencies);
      for (auto &DepInfo : DependsInfo.List) {
        Instruction *NumDependenciesLoad = IRB.CreateLoad(
            NumDependenciesTy, NumDependencies);
        Value *NumDependenciesIncr = IRB.CreateAdd(NumDependenciesLoad, IRB.getInt64(1));
        Instruction *NumDependenciesStore = IRB.CreateStore(NumDependenciesIncr, NumDependencies);
        if (const auto *MultiDepInfo = dyn_cast<MultiDependInfo>(DepInfo.get())) {

          // Build a BasicBlock containing the num_deps increment
          NumDependenciesLoad->getParent()->splitBasicBlock(NumDependenciesLoad);
          Instruction *AfterNumDependenciesStore = NumDependenciesStore->getNextNode();
          AfterNumDependenciesStore->getParent()->splitBasicBlock(AfterNumDependenciesStore);

          // NOTE: after spliting IRBuilder is pointing to a bad BasicBlock.
          // Set again the insert point
          IRB.SetInsertPoint(AfterNumDependenciesStore);

          buildLoopForMultiDep(NumDependenciesLoad, NumDependenciesStore, MultiDepInfo);
        }
      }
    }

    // Arguments for creating a task or a loop directive
    SmallVector<Value *, 4> CreateDirectiveArgs = {
        TaskInfoVar,
        TaskInvInfoVar,
        DirEnv.InstanceLabel
          ? DirEnv.InstanceLabel
          : Constant::getNullValue(PtrTy),
        TaskArgsSizeOf,
        TaskArgsVar,
        TaskPtrVar,
        TaskFlagsVar,
        IRB.CreateLoad(NumDependenciesTy, NumDependencies)
    };

    if (DirEnv.isOmpSsLoopDirective()) {
      SmallVector<Value *> NormalizedUBs(LoopInfo.UBound.size());
      ComputeLoopBounds(&*IRB.saveIP().getPoint(), NormalizedUBs);

      Value *Niters = ConstantInt::get(nanos6Api::Nanos6LoopBounds::getInstance(M).getUBType(), 1);
      for (size_t i = 0; i < LoopInfo.LBound.size(); ++i)
        Niters = IRB.CreateMul(Niters, NormalizedUBs[i]);

      Value *RegisterGrainsize =
        ConstantInt::get(
          nanos6Api::Nanos6LoopBounds::getInstance(M).getGrainsizeType(), 0);
      if (LoopInfo.Grainsize)
        RegisterGrainsize = LoopInfo.Grainsize;

      Value *RegisterChunksize =
        ConstantInt::get(
          nanos6Api::Nanos6LoopBounds::getInstance(M).getChunksizeType(), 0);
      if (LoopInfo.Chunksize)
        RegisterChunksize = LoopInfo.Chunksize;

      Value *RegisterLowerB = ConstantInt::get(nanos6Api::Nanos6LoopBounds::getInstance(M).getLBType(), 0);
      CreateDirectiveArgs.push_back(RegisterLowerB);
      CreateDirectiveArgs.push_back(Niters);
      CreateDirectiveArgs.push_back(
        createZSExtOrTrunc(
          IRB, RegisterGrainsize,
          nanos6Api::Nanos6LoopBounds::getInstance(M).getGrainsizeType(), /*Signed=*/false));
      CreateDirectiveArgs.push_back(
        createZSExtOrTrunc(
          IRB, RegisterChunksize,
          nanos6Api::Nanos6LoopBounds::getInstance(M).getChunksizeType(), /*Signed=*/false));

      IRB.CreateCall(nanos6Api::createLoopFuncCallee(M), CreateDirectiveArgs);
    } else if (DirEnv.isOmpSsTaskIterDirective()) {
      Value *Niters = ConstantInt::get(nanos6Api::Nanos6LoopBounds::getInstance(M).getUBType(), 1);
      if (DirEnv.isOmpSsTaskIterForDirective()) {
        SmallVector<Value *> NormalizedUBs(LoopInfo.UBound.size());
        ComputeLoopBounds(&*IRB.saveIP().getPoint(), NormalizedUBs);

        for (size_t i = 0; i < LoopInfo.LBound.size(); ++i)
          Niters = IRB.CreateMul(Niters, NormalizedUBs[i]);
      }

      Value *RegisterUnroll =
        ConstantInt::get(
          nanos6Api::createIterFuncCallee(M).getFunctionType()->getParamType(9), 0);
      if (LoopInfo.Unroll)
        RegisterUnroll = LoopInfo.Unroll;

      Value *RegisterLowerB = ConstantInt::get(nanos6Api::Nanos6LoopBounds::getInstance(M).getLBType(), 0);
      CreateDirectiveArgs.push_back(RegisterLowerB);
      CreateDirectiveArgs.push_back(Niters);
      CreateDirectiveArgs.push_back(
        createZSExtOrTrunc(
          IRB, RegisterUnroll,
          nanos6Api::createIterFuncCallee(M).getFunctionType()->getParamType(9), /*Signed=*/false));

      IRB.CreateCall(nanos6Api::createIterFuncCallee(M), CreateDirectiveArgs);
    } else {
      IRB.CreateCall(nanos6Api::createTaskFuncCallee(M), CreateDirectiveArgs);
    }

    // DSA capture
    Value *TaskArgsVarL = IRB.CreateLoad(
        PtrTy, TaskArgsVar);

    Value *TaskArgsVarLi8IdxGEP =
      IRB.CreateGEP(Int8Ty, TaskArgsVarL, TaskArgsStructSizeOf, "args_end");

    SmallVector<VLAAlign, 2> VLAAlignsInfo;
    computeVLAsAlignOrder(VLAAlignsInfo);

    if (!DeviceInfo.empty()) {
      int DevGEPIdx = 0;
      // Add device info
      Value *Idx[2], *GEP;
      Idx[0] = Constant::getNullValue(Int32Ty);
      // size_t global_size0;
      // ...
      // size_t global_sizeN-1;

      // size_t local_size0;
      // ...
      // size_t local_sizeN-1;
      const size_t PartNdrangeLength = 3;
      const size_t NdrangeLength =
        DeviceInfo.HasLocalSize ? DeviceInfo.Ndrange.size()/2 : DeviceInfo.Ndrange.size();
      for (size_t j = 0; j < 2; ++j) {
        for (size_t i = 0; i < PartNdrangeLength; ++i) {
          Idx[1] = ConstantInt::get(Int32Ty, DevGEPIdx++);
          GEP = IRB.CreateGEP(
              TaskArgsTy,
              TaskArgsVarL, Idx, ("gep_dev_ndrange" + Twine(i)).str());
          Value *V = ConstantInt::get(Int64Ty, -1);
          if (j <= DeviceInfo.HasLocalSize && i < NdrangeLength)
            V = createZSExtOrTrunc(
              IRB, DeviceInfo.Ndrange[NdrangeLength*j + i],
              Int64Ty, /*Signed=*/false);

          IRB.CreateStore(V, GEP);
        }
      }
      // size_t shm_size;
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, DevGEPIdx++);
      GEP = IRB.CreateGEP(
          TaskArgsTy,
          TaskArgsVarL, Idx, "gep_dev_shm");
      if (DeviceInfo.Shmem) {
        IRB.CreateStore(
          IRB.CreateZExtOrTrunc(DeviceInfo.Shmem, Int64Ty), GEP);
      } else {
        IRB.CreateStore(
          ConstantInt::get(Int64Ty, 0), GEP);
      }
    }

    // First point VLAs to its according space in task args
    for (const auto& VAlign : VLAAlignsInfo) {
      auto *V = VAlign.V;
      Type *Ty = DirEnv.getDSAType(V);
      Align TyAlign = VAlign.TyAlign;

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, TaskArgsToStructIdxMap[V]);
      Value *GEP =
          IRB.CreateGEP(TaskArgsTy,
                        TaskArgsVarL, Idx, "gep_" + V->getName());

      // Point VLA in task args to an aligned position of the extra space allocated
      IRB.CreateAlignedStore(TaskArgsVarLi8IdxGEP, GEP, TyAlign);
      // Skip current VLA size
      unsigned SizeB = DL.getTypeAllocSize(Ty);
      Value *VLASize = ConstantInt::get(Int64Ty, SizeB);
      for (auto *Dim : VLADimsInfo.lookup(V))
        VLASize = IRB.CreateNUWMul(VLASize, Dim);
      TaskArgsVarLi8IdxGEP = IRB.CreateGEP(Int8Ty, TaskArgsVarLi8IdxGEP, VLASize);
    }

    for (const auto &Pair : DSAInfo.Shared) {
      Value *V = Pair.first;
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, TaskArgsToStructIdxMap[V]);
      Value *GEP = IRB.CreateGEP(
          TaskArgsTy,
          TaskArgsVarL, Idx, "gep_" + V->getName());
      IRB.CreateStore(V, GEP);
    }
    for (const auto &Pair : DSAInfo.Private) {
      Value *V = Pair.first;
      Type *Ty = Pair.second;
      // Call custom constructor generated in clang in non-pods
      // Leave pods unititialized
      auto It = DirEnv.NonPODsInfo.Inits.find(V);
      if (It != DirEnv.NonPODsInfo.Inits.end()) {
        // Compute num elements
        Value *NSize = ConstantInt::get(Int64Ty, 1);
        if (isa<ArrayType>(Ty)) {
          while (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
            // Constant array
            Value *NumElems = ConstantInt::get(Int64Ty, ArrTy->getNumElements());
            NSize = IRB.CreateNUWMul(NSize, NumElems);
            Ty = ArrTy->getElementType();
          }
        } else if (VLADimsInfo.count(V)) {
          for (auto *Dim : VLADimsInfo.lookup(V))
            NSize = IRB.CreateNUWMul(NSize, Dim);
        }

        Value *Idx[2];
        Idx[0] = Constant::getNullValue(Int32Ty);
        Idx[1] = ConstantInt::get(Int32Ty, TaskArgsToStructIdxMap[V]);
        Value *GEP = IRB.CreateGEP(
            TaskArgsTy,
            TaskArgsVarL, Idx, "gep_" + V->getName());

        // VLAs
        if (VLADimsInfo.count(V))
          GEP = IRB.CreateLoad(PtrTy, GEP);

        IRB.CreateCall(FunctionCallee(cast<Function>(It->second)), ArrayRef<Value*>{GEP, NSize});
      }
    }
    for (const auto &Pair : DSAInfo.Firstprivate) {
      Value *V = Pair.first;
      Type *Ty = Pair.second;
      Align TyAlign = DL.getPrefTypeAlign(Ty);

      // Compute num elements
      Value *NSize = ConstantInt::get(Int64Ty, 1);
      if (isa<ArrayType>(Ty)) {
        while (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
          // Constant array
          Value *NumElems = ConstantInt::get(Int64Ty, ArrTy->getNumElements());
          NSize = IRB.CreateNUWMul(NSize, NumElems);
          Ty = ArrTy->getElementType();
        }
      } else if (VLADimsInfo.count(V)) {
        for (auto *Dim : VLADimsInfo.lookup(V))
          NSize = IRB.CreateNUWMul(NSize, Dim);
      }

      // call custom copy constructor generated in clang in non-pods
      // do a memcpy if pod
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, TaskArgsToStructIdxMap[V]);
      Value *GEP = IRB.CreateGEP(
          TaskArgsTy,
          TaskArgsVarL, Idx, "gep_" + V->getName());

      // VLAs
      if (VLADimsInfo.count(V))
        GEP = IRB.CreateLoad(PtrTy, GEP);

      auto It = DirEnv.NonPODsInfo.Copies.find(V);
      if (It != DirEnv.NonPODsInfo.Copies.end()) {
        // Non-POD
        llvm::Function *Func = cast<Function>(It->second);
        IRB.CreateCall(Func, ArrayRef<Value*>{/*Src=*/V, /*Dst=*/GEP, NSize});
      } else {
        unsigned SizeB = DL.getTypeAllocSize(Ty);
        Value *NSizeB = IRB.CreateNUWMul(NSize, ConstantInt::get(Int64Ty, SizeB));
        IRB.CreateMemCpy(GEP, TyAlign, V, TyAlign, NSizeB);
      }
    }
    for (Value *V : CapturedInfo) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Int32Ty);
      Idx[1] = ConstantInt::get(Int32Ty, TaskArgsToStructIdxMap[V]);
      Value *GEP = IRB.CreateGEP(
          TaskArgsTy,
          TaskArgsVarL, Idx, "capt_gep_" + V->getName());
      IRB.CreateStore(V, GEP);
    }

    Value *TaskPtrVarL = IRB.CreateLoad(PtrTy, TaskPtrVar);

    CallInst *TaskSubmitFuncCall = IRB.CreateCall(nanos6Api::taskSubmitFuncCallee(M), TaskPtrVarL);
    return TaskSubmitFuncCall;
  };

  void lowerTaskImpl(
      const DirectiveInfo &DirInfo,
      Instruction *CloneEntry, Instruction *CloneExit) {

    Instruction *Entry = DirInfo.Entry;
    Instruction *Exit = DirInfo.Exit;
    if (CloneEntry && CloneExit) {
      Entry = CloneEntry;
      Exit = CloneExit;
    }

    DebugLoc DLoc = Entry->getDebugLoc();
    unsigned Line = 0;
    unsigned Col = 0;
    if (DLoc) {
      Line = DLoc.getLine();
      Col = DLoc.getCol();
    }
    std::string FileNamePlusLoc = (M.getSourceFileName()
                                   + ":" + Twine(Line)
                                   + ":" + Twine(Col)).str();

    Constant *Nanos6TaskLocStr = IRBuilder<>(Entry).CreateGlobalStringPtr(FileNamePlusLoc);
    Constant *Nanos6TaskDeclSourceStr = nullptr;
    if (!DirEnv.DeclSourceStringRef.empty())
      Nanos6TaskDeclSourceStr = IRBuilder<>(Entry).CreateGlobalStringPtr(DirEnv.DeclSourceStringRef);
    Constant *Nanos6TaskDevFuncStr = nullptr;
    if (!DirEnv.DeviceInfo.DevFuncStringRef.empty())
      Nanos6TaskDevFuncStr = IRBuilder<>(Entry).CreateGlobalStringPtr(DirEnv.DeviceInfo.DevFuncStringRef);

    // In loop constructs this will be the starting loop BB
    Instruction *NewEntryI = Entry;
    Instruction *NewExitI = &Exit->getParent()->getUniqueSuccessor()->front();

    // Create nanos6_task_args_* START
    SmallVector<Type *, 4> TaskArgsMemberTy;
    MapVector<Value *, size_t> TaskArgsToStructIdxMap;
    StructType *TaskArgsTy = createTaskArgsType(
      TaskArgsToStructIdxMap, ("nanos6_task_args_" + F.getName()).str());
    // Create nanos6_task_args_* END

    SmallVector<Type *, 4> TaskTypeList;
    SmallVector<StringRef, 4> TaskNameList;
    GlobalVariable *SizeofTableVar = nullptr;
    GlobalVariable *OffsetTableVar = nullptr;
    GlobalVariable *ArgIdxTableVar = nullptr;

    getTaskArgsInfo(
      TaskArgsToStructIdxMap, TaskArgsTy, TaskTypeList, TaskNameList,
      SizeofTableVar, OffsetTableVar, ArgIdxTableVar);

    Function *OlDestroyArgsFuncVar =
      createDestroyArgsOlFunc(TaskArgsToStructIdxMap, TaskArgsTy, TaskTypeList, TaskNameList);

    Function *OlDuplicateArgsFuncVar =
      createDuplicateArgsOlFunc(TaskArgsToStructIdxMap, TaskArgsTy, TaskTypeList, TaskNameList);

    Function *OlTaskFuncVar = nullptr;
    Function *UnpackTaskFuncVar = nullptr;
      createTaskFuncOl(
        TaskArgsToStructIdxMap, TaskArgsTy,
        TaskTypeList, TaskNameList, DirEnv.isOmpSsLoopDirective(), OlTaskFuncVar, UnpackTaskFuncVar);

    Function *OlDepsFuncVar =
      createDepsOlFunc(TaskArgsToStructIdxMap, TaskArgsTy, TaskTypeList, TaskNameList);

    Function *OlConstraintsFuncVar =
      createConstraintsOlFunc(TaskArgsToStructIdxMap, TaskArgsTy, TaskTypeList, TaskNameList);

    Function *OlPriorityFuncVar =
      createPriorityOlFunc(TaskArgsToStructIdxMap, TaskArgsTy, TaskTypeList, TaskNameList);

    Function *OlOnreadyFuncVar =
      createOnreadyOlFunc(TaskArgsToStructIdxMap, TaskArgsTy, TaskTypeList, TaskNameList);

    Function *OlTaskIterWhileCondFuncVar =
      createTaskIterWhileCondOlFunc(TaskArgsToStructIdxMap, TaskArgsTy, TaskTypeList, TaskNameList);

    DirectiveLoopInfo NewLoopInfo = LoopInfo;
    SmallVector<Value *> NormalizedUBs(LoopInfo.UBound.size());
    SmallVector<Instruction *> CollapseIterBB;
    if (DirEnv.isOmpSsLoopDirective())
      buildUnpackedLoopForTask(NewLoopInfo, NewEntryI, NewExitI, NormalizedUBs, CollapseIterBB);
    if (DirEnv.isOmpSsTaskIterForDirective()) {
      Type *OrigIndVarTy = DirEnv.getDSAType(LoopInfo.IndVar[0]);
      // Increment induction variable at the end of the taskiter for
      IRBuilder<> IRB(DirInfo.Exit);
      Instruction *IndVarVal = IRB.CreateLoad(OrigIndVarTy, LoopInfo.IndVar[0]);
      Instruction *StepResult = IRB.CreateCall(LoopInfo.Step[0].Fun, LoopInfo.Step[0].Args);
      auto p = buildAddSignDependent(
        IRB, IndVarVal, StepResult, OrigIndVarTy, LoopInfo.StepSigned[0]);
      IRB.CreateStore(createZSExtOrTrunc(IRB, p.first, OrigIndVarTy, p.second), LoopInfo.IndVar[0]);
    }

    SetVector<Instruction *> TaskBBs;
    computeBBsBetweenEntryExit(TaskBBs, NewEntryI, NewExitI);

    // 3. Create Nanos6 task data structures info
    GlobalVariable *TaskInvInfoVar =
      new GlobalVariable(M, nanos6Api::Nanos6TaskInvInfo::getInstance(M).getType(),
        /*isConstant=*/true, GlobalVariable::InternalLinkage,
        ConstantStruct::get(nanos6Api::Nanos6TaskInvInfo::getInstance(M).getType(),
          Nanos6TaskLocStr),
        ("task_invocation_info_" + F.getName()).str());
    TaskInvInfoVar->setAlignment(Align(64));

    GlobalVariable *TaskImplInfoVar =
      new GlobalVariable(M, ArrayType::get(nanos6Api::Nanos6TaskImplInfo::getInstance(M).getType(), 1),
        /*isConstant=*/true, GlobalVariable::InternalLinkage,
        ConstantArray::get(ArrayType::get(nanos6Api::Nanos6TaskImplInfo::getInstance(M).getType(), 1), // TODO: More than one implementations?
        ConstantStruct::get(nanos6Api::Nanos6TaskImplInfo::getInstance(M).getType(),
        DirEnv.DeviceInfo.Kind
          ? cast<Constant>(DirEnv.DeviceInfo.Kind)
          : ConstantInt::get(nanos6Api::Nanos6TaskImplInfo::getInstance(M).getDeviceTypeIdType(), 0),
        cast<Constant>(OlTaskFuncVar),
        OlConstraintsFuncVar ? cast<Constant>(OlConstraintsFuncVar) : ConstantPointerNull::get(PtrTy),
        DirEnv.Label ? cast<Constant>(DirEnv.Label) : ConstantPointerNull::get(PtrTy),
        Nanos6TaskDeclSourceStr ? Nanos6TaskDeclSourceStr : Nanos6TaskLocStr,
        Nanos6TaskDevFuncStr
          ? Nanos6TaskDevFuncStr
          : ConstantPointerNull::get(PtrTy))),
        ("implementations_var_" + F.getName()).str());
    TaskImplInfoVar->setAlignment(Align(64));

    SmallVector<Constant *, 4> Inits;
    for (auto &p : DirEnv.ReductionsInitCombInfo)
      Inits.push_back(cast<Constant>(p.second.Init));
    GlobalVariable *TaskRedInitsVar =
      new GlobalVariable(M,
        ArrayType::get(PtrTy, DirEnv.ReductionsInitCombInfo.size()),
        /*isConstant=*/true, GlobalVariable::InternalLinkage,
        ConstantArray::get(ArrayType::get(
          PtrTy, DirEnv.ReductionsInitCombInfo.size()), Inits),
        ("nanos6_reduction_initializers_" + F.getName()).str());
    TaskRedInitsVar->setAlignment(Align(64));

    SmallVector<Constant *, 4> Combs;
    for (auto &p : DirEnv.ReductionsInitCombInfo)
      Combs.push_back(cast<Constant>(p.second.Comb));
    GlobalVariable *TaskRedCombsVar =
      new GlobalVariable(M,
        ArrayType::get(PtrTy, DirEnv.ReductionsInitCombInfo.size()),
        /*isConstant=*/true, GlobalVariable::InternalLinkage,
        ConstantArray::get(ArrayType::get(
          PtrTy, DirEnv.ReductionsInitCombInfo.size()), Combs),
        ("nanos6_reduction_combiners_" + F.getName()).str());

    GlobalVariable *TaskInfoVar =
      new GlobalVariable(M, nanos6Api::Nanos6TaskInfo::getInstance(M).getType(),
        /*isConstant=*/false, // TaskInfo is modified by nanos6
        GlobalVariable::InternalLinkage,
        ConstantStruct::get(nanos6Api::Nanos6TaskInfo::getInstance(M).getType(),
          ConstantInt::get(nanos6Api::Nanos6TaskInfo::getInstance(M).getNumSymbolsType(), DirEnv.DependsInfo.NumSymbols),
          OlDepsFuncVar ? cast<Constant>(OlDepsFuncVar) : ConstantPointerNull::get(PtrTy),
          OlOnreadyFuncVar ? cast<Constant>(OlOnreadyFuncVar) : ConstantPointerNull::get(PtrTy),
          OlPriorityFuncVar ? cast<Constant>(OlPriorityFuncVar) : ConstantPointerNull::get(PtrTy),
          ConstantInt::get(nanos6Api::Nanos6TaskInfo::getInstance(M).getImplCountType(), 1),
          TaskImplInfoVar,
          OlDestroyArgsFuncVar ? cast<Constant>(OlDestroyArgsFuncVar) : ConstantPointerNull::get(PtrTy),
          OlDuplicateArgsFuncVar ? cast<Constant>(OlDuplicateArgsFuncVar) : ConstantPointerNull::get(PtrTy),
          TaskRedInitsVar,
          TaskRedCombsVar,
          ConstantPointerNull::get(PtrTy),
          OlTaskIterWhileCondFuncVar ? cast<Constant>(OlTaskIterWhileCondFuncVar) : ConstantPointerNull::get(PtrTy),
          ConstantInt::get(nanos6Api::Nanos6TaskInfo::getInstance(M).getNumArgsType(), TaskTypeList.size()),
          SizeofTableVar,
          OffsetTableVar,
          ArgIdxTableVar),
        ("task_info_var_" + F.getName()).str());
    TaskInfoVar->setAlignment(Align(64));

    registerTaskInfo(TaskInfoVar);

    // FIXME: duplicated from analysis valueInDSABundles,
    // but needed in CodeExtractor
    auto valueInUnpackParams = [&TaskArgsToStructIdxMap](Value *const V) {
      int ret = -1;
      if (TaskArgsToStructIdxMap.count(V))
        ret = TaskArgsToStructIdxMap.lookup(V);
      return ret;
    };

    // 4. Extract region the way we want
    CodeExtractorAnalysisCache CEAC(F);
    SmallVector<BasicBlock *> TaskBBs1;
    for (auto *I : TaskBBs)
      TaskBBs1.push_back(I->getParent());

    auto fwdRewriteUsesBrAndGetOmpSsUnpackFunc = std::bind(
      &OmpSsDirective::rewriteUsesBrAndGetOmpSsUnpackFunc, this, NewLoopInfo, NormalizedUBs,
      CollapseIterBB, UnpackTaskFuncVar, TaskArgsToStructIdxMap, DirInfo.Exit,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
      std::placeholders::_4, std::placeholders::_5);
    auto fwdEmitOmpSsCaptureAndSubmitTask = std::bind(
      &OmpSsDirective::emitOmpSsCaptureAndSubmitTask, this, DLoc, TaskArgsTy,
      TaskArgsToStructIdxMap, TaskInfoVar, TaskInvInfoVar,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    CodeExtractor CE(TaskBBs1, fwdRewriteUsesBrAndGetOmpSsUnpackFunc, fwdEmitOmpSsCaptureAndSubmitTask, valueInUnpackParams);
    CE.extractCodeRegion(CEAC);
  }

  void lowerTask() {

    Instruction *CloneEntry = nullptr;

    lowerTaskImpl(
      DirInfo, CloneEntry, CloneEntry);

    DirInfo.Exit->eraseFromParent();
    DirInfo.Entry->eraseFromParent();
  }

  void lowerCritical() {
    IRBuilder<> IRB(DirInfo.Entry);
    unsigned Line = DirInfo.Entry->getDebugLoc().getLine();
    unsigned Col = DirInfo.Entry->getDebugLoc().getCol();

    std::string FileNamePlusLoc = (M.getSourceFileName()
                                   + ":" + Twine(Line)
                                   + ":" + Twine(Col)).str();
    Constant *Nanos6CriticalLocStr = IRB.CreateGlobalStringPtr(FileNamePlusLoc);

    GlobalVariable *GLock = cast<GlobalVariable>(
      M.getOrInsertGlobal(DirEnv.CriticalNameStringRef, PtrTy));
    GLock->setLinkage(GlobalValue::WeakAnyLinkage);
    GLock->setInitializer(Constant::getNullValue(PtrTy));

    if (DirEnv.isOmpSsCriticalStartDirective()) {
      IRB.CreateCall(nanos6Api::userLockFuncCallee(M), {GLock, Nanos6CriticalLocStr});
    } else {
      IRB.CreateCall(nanos6Api::userUnlockFuncCallee(M), {GLock});
    }

    DirInfo.Entry->eraseFromParent();
  }

  void lowerFinalCode() {
    // Skip non task directives
    if (!DirEnv.isOmpSsTaskDirective())
      return;

    InFinalRAIIObject InFinalRAII(*this);

    if (DirEnv.isOmpSsTaskDirective())
      buildFinalCondCFG();

    // Erase cloned intrinsics
    if (DirEnv.isOmpSsTaskDirective()) {
      Instruction *CloneEntryI = cast<Instruction>(FinalInfo.VMap.lookup(DirInfo.Entry));
      Instruction *CloneExitI = cast<Instruction>(FinalInfo.VMap.lookup(DirInfo.Exit));

      bool IsDeviceWithNdrange =
        !DirEnv.DeviceInfo.empty()
          && !DirEnv.DeviceInfo.Ndrange.empty();
      if (IsDeviceWithNdrange)
        lowerTaskImpl(DirInfo, CloneEntryI, CloneExitI);

      CloneExitI->eraseFromParent();
      CloneEntryI->eraseFromParent();
      for (size_t j = 0; j < DirInfo.InnerDirectiveInfos.size(); ++j) {
        DirectiveInfo &InnerDirInfo = *DirInfo.InnerDirectiveInfos[j];

        CloneEntryI = FinalInfo.InnerClonedEntries[j];
        CloneExitI = FinalInfo.InnerClonedExits[j];

        bool IsDeviceWithNdrange =
          !InnerDirInfo.DirEnv.DeviceInfo.empty()
            && !InnerDirInfo.DirEnv.DeviceInfo.Ndrange.empty();
        if (IsDeviceWithNdrange)
          lowerTaskImpl(InnerDirInfo, CloneEntryI, CloneExitI);

        CloneExitI->eraseFromParent();
        CloneEntryI->eraseFromParent();
      }
    }
  }

  // This must be called before erasing original entry/exit
  void buildFinalCondCFG() {
    // Lower final inner tasks

    SmallVector<Instruction *, 4> Allocas;

    // Process all the inner directives before
    for (size_t i = 0; i < DirInfo.InnerDirectiveInfos.size(); ++i) {
      const DirectiveInfo &InnerDirInfo = *DirInfo.InnerDirectiveInfos[i];
      // Build loop for taskloop/taskfor taskloopfor taskiterfor
      bool IsLoop =
        InnerDirInfo.DirEnv.isOmpSsLoopDirective() ||
        InnerDirInfo.DirEnv.isOmpSsTaskIterForDirective();
      bool IsWhile = InnerDirInfo.DirEnv.isOmpSsTaskIterWhileDirective();
      Instruction *CloneEntryI = FinalInfo.InnerClonedEntries[i];
      Instruction *CloneExitI = FinalInfo.InnerClonedExits[i];

      gatherConstAllocas(CloneEntryI);

      if (IsLoop) {
        DirectiveLoopInfo FinalLoopInfo = InnerDirInfo.DirEnv.LoopInfo;
        rewriteDirInfoForFinal(FinalLoopInfo, FinalInfo.VMap);
        buildLoopForTask(InnerDirInfo.DirEnv, CloneEntryI, CloneExitI, FinalLoopInfo);
      } else if (IsWhile) {
        DirectiveWhileInfo FinalWhileInfo = InnerDirInfo.DirEnv.WhileInfo;
        rewriteDirInfoForFinal(FinalWhileInfo, FinalInfo.VMap);
        buildWhileForTask(CloneEntryI, CloneExitI, FinalWhileInfo);
      }
    }
    bool IsLoop =
      DirInfo.DirEnv.isOmpSsLoopDirective() ||
      DirInfo.DirEnv.isOmpSsTaskIterForDirective();
    bool IsWhile = DirInfo.DirEnv.isOmpSsTaskIterWhileDirective();
    Instruction *OrigEntryI = DirInfo.Entry;
    Instruction *OrigExitI = DirInfo.Exit;
    Instruction *CloneEntryI = cast<Instruction>(FinalInfo.VMap.lookup(DirInfo.Entry));
    Instruction *CloneExitI = cast<Instruction>(FinalInfo.VMap.lookup(DirInfo.Exit));

    gatherConstAllocas(CloneEntryI);
    gatherConstAllocas(OrigEntryI);

    Instruction *NewCloneEntryI = CloneEntryI;
    if (IsLoop) {
      DirectiveLoopInfo FinalLoopInfo = DirInfo.DirEnv.LoopInfo;
      rewriteDirInfoForFinal(FinalLoopInfo, FinalInfo.VMap);
      NewCloneEntryI = buildLoopForTask(DirInfo.DirEnv, CloneEntryI, CloneExitI, FinalLoopInfo);
    } else if (IsWhile) {
      DirectiveWhileInfo FinalWhileInfo = DirInfo.DirEnv.WhileInfo;
      rewriteDirInfoForFinal(FinalWhileInfo, FinalInfo.VMap);
      NewCloneEntryI = buildWhileForTask(CloneEntryI, CloneExitI, FinalWhileInfo);
    }

    BasicBlock *OrigEntryBB = OrigEntryI->getParent();
    BasicBlock *OrigExitBB = OrigExitI->getParent()->getUniqueSuccessor();

    OrigExitBB->setName("final.end");
    BasicBlock *FinalCondBB = BasicBlock::Create(Ctx, "final.cond", &F);

    BasicBlock *NewCloneEntryBB = NewCloneEntryI->getParent();
    NewCloneEntryBB->setName("final.then");
    // Move the allocas gathered to the start
    // of the final code
    {
      IRBuilder<> IRB(NewCloneEntryI);
      for (Instruction *I : Allocas) {
        I->removeFromParent();
        IRB.Insert(I, I->getName());
      }
    }

    // We are now just before the branch to task body
    Instruction *EntryBBTerminator = OrigEntryBB->getSinglePredecessor()->getTerminator();

    IRBuilder<> IRB(EntryBBTerminator);

    IRB.CreateBr(FinalCondBB);
    // Remove the old branch
    EntryBBTerminator->eraseFromParent();

    IRB.SetInsertPoint(FinalCondBB);
    // if (nanos6_in_final())
    Value *Cond = IRB.CreateICmpNE(IRB.CreateCall(nanos6Api::taskInFinalFuncCallee(M), {}), IRB.getInt32(0));
    IRB.CreateCondBr(Cond, NewCloneEntryBB, OrigEntryBB);

  }

  bool run() {
    lowerFinalCode();
    if (DirEnv.isOmpSsTaskwaitDirective())
      lowerTaskwait();
    else if (DirEnv.isOmpSsReleaseDirective())
      lowerRelease();
    else if (DirEnv.isOmpSsTaskDirective())
      lowerTask();
    else if (DirEnv.isOmpSsCriticalDirective())
      lowerCritical();

    return true;
  }
};

struct OmpSsFunction {
  Module &M;
  LLVMContext &Ctx;
  Function &F;
  function_ref<OmpSsRegionAnalysis &(Function &)> LookupDirectiveFunctionInfo;
  // TODO: Avoid using a fixed array
  DirectiveFinalInfo FinalInfos[500];
  SmallVector<Instruction *, 4> PostMoveInstructions;

  OmpSsFunction(Module &M, Function &F,
    function_ref<OmpSsRegionAnalysis &(Function &)> LookupDirectiveFunctionInfo)
      : M(M), Ctx(M.getContext()), F(F),
        LookupDirectiveFunctionInfo(LookupDirectiveFunctionInfo)
        {}

  void relocateInstrs() {
    for (auto *I : PostMoveInstructions) {
      Function *DstF = cast<Function>(I->getParent()->getParent());
      Instruction *TI = DstF->getEntryBlock().getTerminator();
      I->moveBefore(TI);
    }
  }

  void buildFinalCloneBBs(const DirectiveFunctionInfo &DirectiveFuncInfo) {
    if (DirectiveFuncInfo.PostOrder.size() > 500)
      llvm_unreachable("Exceeded final info elements");

    // First sweep to clone BBs
    for (size_t i = 0; i < DirectiveFuncInfo.PostOrder.size(); ++i) {
      DirectiveInfo &DirInfo = *DirectiveFuncInfo.PostOrder[i];

      // Skip non task directives
      if (!DirInfo.DirEnv.isOmpSsTaskDirective())
        continue;

      // 1. Split BB
      BasicBlock *EntryBB = DirInfo.Entry->getParent();
      EntryBB = EntryBB->splitBasicBlock(DirInfo.Entry);

      BasicBlock *ExitBB = DirInfo.Exit->getParent();
      ExitBB = ExitBB->splitBasicBlock(DirInfo.Exit->getNextNode());

      SmallVector<BasicBlock*, 8> Worklist;
      SmallPtrSet<BasicBlock*, 8> Visited;
      DenseMap<BasicBlock *, BasicBlock *> CopyBBs;
      ValueToValueMapTy &VMap = FinalInfos[i].VMap;

      // 2. Clone BBs between entry and exit (is there any function/util to do this?)
      Worklist.push_back(EntryBB);
      Visited.insert(EntryBB);

      CopyBBs[EntryBB] = CloneBasicBlock(EntryBB, VMap, ".clone", &F);
      VMap[EntryBB] = CopyBBs[EntryBB];
      while (!Worklist.empty()) {
        auto WIt = Worklist.begin();
        BasicBlock *BB = *WIt;
        Worklist.erase(WIt);

        for (auto It = succ_begin(BB); It != succ_end(BB); ++It) {
          if (!Visited.count(*It) && *It != ExitBB) {
            Worklist.push_back(*It);
            Visited.insert(*It);

            CopyBBs[*It] = CloneBasicBlock(*It, VMap, ".clone", &F);
            VMap[*It] = CopyBBs[*It];
          }
        }
      }

      // 2. Rewrite ops and branches to cloned ones.
      //    Intrinsic exit is mapped to the original entry, so before removing it
      //    we must to map it to the cloned entry.
      for (auto &p : CopyBBs) {
        BasicBlock *& CopyBB = p.second;
        for (BasicBlock::iterator II = CopyBB->begin(), E = CopyBB->end(); II != E;) {
          Instruction &I = *II++;
          // Remove OmpSs-2 intrinsics directive_marker. Cloned entries/exits will be removed
          // when building final stuff
          if (auto *IIntr = dyn_cast<IntrinsicInst>(&I)) {
            Intrinsic::ID IID = IIntr->getIntrinsicID();
            if (IID == Intrinsic::directive_marker) {
              IIntr->eraseFromParent();
              continue;
            }
          }
          RemapInstruction(&I, VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
        }
      }
      // Gather all cloned Entry/Exit of inner directives to pass them
      // to OmpSsDirective
      for (size_t j = 0; j < DirInfo.InnerDirectiveInfos.size(); ++j) {
        const DirectiveInfo &InnerDirInfo = *DirInfo.InnerDirectiveInfos[j];
        Instruction *OrigEntryI = InnerDirInfo.Entry;
        Instruction *OrigExitI = InnerDirInfo.Exit;
        Instruction *CloneEntryI = cast<Instruction>(VMap.lookup(OrigEntryI));
        Instruction *CloneExitI = cast<Instruction>(VMap.lookup(OrigExitI));
        FinalInfos[i].InnerClonedEntries.push_back(CloneEntryI);
        FinalInfos[i].InnerClonedExits.push_back(CloneExitI);
      }
    }
  }

  bool run() {
    // Nothing to do for declarations.
    if (F.isDeclaration() || F.empty())
      return false;

    DirectiveFunctionInfo &DirectiveFuncInfo = LookupDirectiveFunctionInfo(F).getFuncInfo();

    // Emit check version call if translation unit has at least one ompss-2
    // directive
    if (!DirectiveFuncInfo.PostOrder.empty())
      registerCheckVersion(M);

    buildFinalCloneBBs(DirectiveFuncInfo);
    for (size_t i = 0; i < DirectiveFuncInfo.PostOrder.size(); ++i) {
      DirectiveInfo &DirInfo = *DirectiveFuncInfo.PostOrder[i];
      OmpSsDirective(M, F, DirInfo, FinalInfos[i], PostMoveInstructions).run();
    }
    relocateInstrs();
    return !DirectiveFuncInfo.PostOrder.empty();
  }
};

struct OmpSsModule {
  Module &M;
  LLVMContext &Ctx;
  OmpSsModule(Module &M,
    function_ref<OmpSsRegionAnalysis &(Function &)> LookupDirectiveFunctionInfo)
      : M(M), Ctx(M.getContext()), LookupDirectiveFunctionInfo(LookupDirectiveFunctionInfo)
        {}
  function_ref<OmpSsRegionAnalysis &(Function &)> LookupDirectiveFunctionInfo;

  void registerAssert(StringRef Str) {
    // Emit check version call if translation unit has at least one ompss-2
    // directive
    registerCheckVersion(M);
    Function *Func = cast<Function>(nanos6Api::registerCtorAssertFuncCallee(M).getCallee());
    if (Func->empty()) {
      Func->setLinkage(GlobalValue::InternalLinkage);
      BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", Func);
      Instruction *RetInst = ReturnInst::Create(Ctx);
      RetInst->insertInto(EntryBB, EntryBB->end());

      appendToGlobalCtors(M, Func, 65535);

    }
    BasicBlock &Entry = Func->getEntryBlock();

    IRBuilder<> BBBuilder(&Entry.back());
    Constant *StringPtr = BBBuilder.CreateGlobalStringPtr(Str);
    BBBuilder.CreateCall(nanos6Api::registerAssertFuncCallee(M), StringPtr);
  }

  bool run() {
    if (M.empty())
      return false;

    // Try to find asserts and add them to the nanos6_constructor_register_assert()
    const NamedMDNode *ModuleMD = M.getModuleFlagsMetadata();
    if (ModuleMD) {
      for (const MDNode *ModuleMDOp : ModuleMD->operands()) {
        assert(ModuleMDOp->getNumOperands() == 3);
        StringRef IDStrRef = cast<MDString>(ModuleMDOp->getOperand(1))->getString();
        if (IDStrRef == "OmpSs-2 Metadata") {
          const MDNode *OssMD = cast<MDNode>(ModuleMDOp->getOperand(2));
          for (const MDOperand &OssMDOp : OssMD->operands()) {
            const MDNode *Op = cast<MDNode>(OssMDOp.get());
            assert(Op->getNumOperands() == 2);
            StringRef KeyStrRef = cast<MDString>(Op->getOperand(0))->getString();
            StringRef ValueStrRef = cast<MDString>(Op->getOperand(1))->getString();
            if (KeyStrRef == "assert") {
              registerAssert(ValueStrRef);
            }
          }
        }
      }
    }

    bool Modified = false;
    for (auto &F : M)
      Modified |= OmpSsFunction(M, F, LookupDirectiveFunctionInfo).run();
    return Modified;
  }
};

struct OmpSsLegacyPass : public ModulePass {
  /// Pass identification, replacement for typeid
  static char ID;
  OmpSsLegacyPass() : ModulePass(ID) {
    initializeOmpSsLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;
    auto LookupDirectiveFunctionInfo = [this](Function &F) -> OmpSsRegionAnalysis & {
      return this->getAnalysis<OmpSsRegionAnalysisLegacyPass>(F).getResult();
    };
    return OmpSsModule(M, LookupDirectiveFunctionInfo).run();
  }

  StringRef getPassName() const override { return "Nanos6 Lowering"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<OmpSsRegionAnalysisLegacyPass>();
  }
};

}

PreservedAnalyses OmpSsPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupDirectiveFunctionInfo = [&FAM](Function &F) -> OmpSsRegionAnalysis & {
    return FAM.getResult<OmpSsRegionAnalysisPass>(F);
  };
  if (!OmpSsModule(M, LookupDirectiveFunctionInfo).run())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

char OmpSsLegacyPass::ID = 0;

ModulePass *llvm::createOmpSsPass() {
  return new OmpSsLegacyPass();
}

void LLVMOmpSsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createOmpSsPass());
}

INITIALIZE_PASS_BEGIN(OmpSsLegacyPass, "ompss-2",
                "Transforms OmpSs-2 llvm.directive.region intrinsics", false, false)
INITIALIZE_PASS_DEPENDENCY(OmpSsRegionAnalysisLegacyPass)
INITIALIZE_PASS_END(OmpSsLegacyPass, "ompss-2",
                "Transforms OmpSs-2 llvm.directive.region intrinsics", false, false)
