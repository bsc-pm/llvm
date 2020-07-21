//===- OmpSs.cpp -- Strip parts of Debug Info --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/OmpSs.h"
#include "llvm/Analysis/OmpSsRegionAnalysis.h"

#include "llvm/Pass.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsOmpSs.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
using namespace llvm;

namespace {

struct OmpSs : public ModulePass {
  /// Pass identification, replacement for typeid
  static char ID;
  OmpSs() : ModulePass(ID) {
    initializeOmpSsPass(*PassRegistry::getPassRegistry());
  }

  bool Initialized = false;

  class Nanos6LoopBounds {
  private:
    StructType *Ty;

    Nanos6LoopBounds(){};
    Nanos6LoopBounds(const Nanos6LoopBounds&){};
  public:
    ~Nanos6LoopBounds(){};

    static auto getInstance(Module &M) -> Nanos6LoopBounds& {
      static auto instance = std::unique_ptr<Nanos6LoopBounds>(nullptr);
      if (!instance) {
        instance.reset(new Nanos6LoopBounds);
        instance->Ty = StructType::create(M.getContext(),
          "nanos6_loop_bounds_t");

        // size_t lower_bound;
        // size_t upper_bound;
        // size_t grainsize;
        // size_t chunksize;

        Type *LBoundTy = Type::getInt64Ty(M.getContext());
        Type *UBoundTy = Type::getInt64Ty(M.getContext());
        Type *GrainsizeTy = Type::getInt64Ty(M.getContext());
        Type *ChunkSizeTy = Type::getInt64Ty(M.getContext());

        instance->Ty->setBody({LBoundTy, UBoundTy, GrainsizeTy, ChunkSizeTy});
      }
      return *instance.get();
    }
    StructType *getType() { return Ty; }
  };

  class Nanos6TaskAddrTranslationEntry {
  private:
    StructType *Ty;

    Nanos6TaskAddrTranslationEntry(){};
    Nanos6TaskAddrTranslationEntry(const Nanos6TaskAddrTranslationEntry&){};
  public:
    ~Nanos6TaskAddrTranslationEntry(){};

    static auto getInstance(Module &M) -> Nanos6TaskAddrTranslationEntry& {
      static auto instance = std::unique_ptr<Nanos6TaskAddrTranslationEntry>(nullptr);
      if (!instance) {
        instance.reset(new Nanos6TaskAddrTranslationEntry);
        instance->Ty = StructType::create(M.getContext(),
          "nanos6_address_translation_entry_t");

        // size_t local_address
        // size_t device_address
        Type *LocalAddrTy = Type::getInt64Ty(M.getContext());
        Type *DeviceAddrTy = Type::getInt64Ty(M.getContext());

        instance->Ty->setBody({LocalAddrTy, DeviceAddrTy});
      }
      return *instance.get();
    }
    StructType *getType() { return Ty; }
  };

  class Nanos6TaskConstraints {
  private:
    StructType *Ty;

    Nanos6TaskConstraints(){};
    Nanos6TaskConstraints(const Nanos6TaskConstraints&){};
  public:
    ~Nanos6TaskConstraints(){};

    static auto getInstance(Module &M) -> Nanos6TaskConstraints& {
      static auto instance = std::unique_ptr<Nanos6TaskConstraints>(nullptr);
      if (!instance) {
        instance.reset(new Nanos6TaskConstraints);
        instance->Ty = StructType::create(M.getContext(),
          "nanos6_task_constraints_t");

        // size_t cost
        Type *CostTy = Type::getInt64Ty(M.getContext());

        instance->Ty->setBody(CostTy);
      }
      return *instance.get();
    }
    StructType *getType() { return Ty; }
  };

  class Nanos6TaskInvInfo {
  private:
    StructType *Ty;

    Nanos6TaskInvInfo(){};
    Nanos6TaskInvInfo(const Nanos6TaskInvInfo&){};
  public:
    ~Nanos6TaskInvInfo(){};

    static auto getInstance(Module &M) -> Nanos6TaskInvInfo& {
      static auto instance = std::unique_ptr<Nanos6TaskInvInfo>(nullptr);
      if (!instance) {
        instance.reset(new Nanos6TaskInvInfo);
        instance->Ty = StructType::create(M.getContext(),
          "nanos6_task_invocation_info_t");

        // const char *invocation_source
        Type *InvSourceTy = Type::getInt8PtrTy(M.getContext());

        instance->Ty->setBody(InvSourceTy);
      }
      return *instance.get();
    }
    StructType *getType() { return Ty; }
  };

  class Nanos6TaskImplInfo {
  private:
    StructType *Ty;

    Nanos6TaskImplInfo(){};
    Nanos6TaskImplInfo(const Nanos6TaskImplInfo&){};
  public:
    ~Nanos6TaskImplInfo(){};

    static auto getInstance(Module &M) -> Nanos6TaskImplInfo& {
      static auto instance = std::unique_ptr<Nanos6TaskImplInfo>(nullptr);
      if (!instance) {
        instance.reset(new Nanos6TaskImplInfo);
        instance->Ty = StructType::create(M.getContext(),
          "nanos6_task_implementation_info_t");

        // int device_type_id;
        Type *DeviceTypeIdTy = Type::getInt32Ty(M.getContext());
        // void (*run)(void *, void *, nanos6_address_translation_entry_t *);
        Type *RunFuncTy =
          FunctionType::get(Type::getVoidTy(M.getContext()),
                            /*IsVarArgs=*/false)->getPointerTo();
        // void (*get_constraints)(void *, nanos6_task_constraints_t *);
        Type *GetConstraintsFuncTy =
          FunctionType::get(Type::getVoidTy(M.getContext()),
                            /*IsVarArgs=*/false)->getPointerTo();
        // const char *task_label;
        Type *TaskLabelTy = Type::getInt8PtrTy(M.getContext());
        // const char *declaration_source;
        Type *DeclSourceTy = Type::getInt8PtrTy(M.getContext());
        // void (*run_wrapper)(void *, void *, nanos6_address_translation_entry_t *);
        Type *RunWrapperFuncTy =
          FunctionType::get(Type::getVoidTy(M.getContext()),
                            /*IsVarArgs=*/false)->getPointerTo();
        instance->Ty->setBody({DeviceTypeIdTy, RunFuncTy,
                              GetConstraintsFuncTy, TaskLabelTy,
                              DeclSourceTy, RunWrapperFuncTy});
      }
      return *instance.get();
    }
    StructType *getType() { return Ty; }
  };

  class Nanos6TaskInfo {
  private:
    StructType *Ty;

    Nanos6TaskInfo(){};
    Nanos6TaskInfo(const Nanos6TaskInfo&){};
  public:
    ~Nanos6TaskInfo(){};

    static auto getInstance(Module &M) -> Nanos6TaskInfo& {
      static auto instance = std::unique_ptr<Nanos6TaskInfo>(nullptr);
      if (!instance) {
        instance.reset(new Nanos6TaskInfo);
        instance->Ty = StructType::create(M.getContext(),
          "nanos6_task_info_t");

        // int num_symbols;
        Type *NumSymbolsTy = Type::getInt32Ty(M.getContext());;
        // void (*register_depinfo)(void *, void *);
        Type *RegisterInfoFuncTy =
          FunctionType::get(Type::getVoidTy(M.getContext()),
                            /*IsVarArgs=*/false)->getPointerTo();
        // void (*get_priority)(void *, nanos6_priority_t *);
        // void (*get_priority)(void *, long int *);
        Type *GetPriorityFuncTy =
          FunctionType::get(Type::getVoidTy(M.getContext()),
                            /*IsVarArgs=*/false)->getPointerTo();
        // int implementation_count;
        Type *ImplCountTy = Type::getInt32Ty(M.getContext());
        // nanos6_task_implementation_info_t *implementations;
        Type *TaskImplInfoTy = StructType::get(M.getContext())->getPointerTo();
        // void (*destroy_args_block)(void *);
        Type *DestroyArgsBlockFuncTy =
          FunctionType::get(Type::getVoidTy(M.getContext()),
                            /*IsVarArgs=*/false)->getPointerTo();
        // void (*duplicate_args_block)(const void *, void **);
        Type *DuplicateArgsBlockFuncTy =
          FunctionType::get(Type::getVoidTy(M.getContext()),
                            /*IsVarArgs=*/false)->getPointerTo();
        // void (**reduction_initializers)(void *, void *, size_t);
        Type *ReductInitsFuncTy =
          FunctionType::get(Type::getVoidTy(M.getContext()),
                            /*IsVarArgs=*/false)->getPointerTo()->getPointerTo();
        // void (**reduction_combiners)(void *, void *, size_t);
        Type *ReductCombsFuncTy =
          FunctionType::get(Type::getVoidTy(M.getContext()),
                            /*IsVarArgs=*/false)->getPointerTo()->getPointerTo();
        // void *task_type_data;
        Type *TaskTypeDataTy =
          Type::getInt8PtrTy(M.getContext());

        instance->Ty->setBody({NumSymbolsTy, RegisterInfoFuncTy, GetPriorityFuncTy,
                               ImplCountTy, TaskImplInfoTy, DestroyArgsBlockFuncTy,
                               DuplicateArgsBlockFuncTy, ReductInitsFuncTy, ReductCombsFuncTy,
                               TaskTypeDataTy
                              });
      }
      return *instance.get();
    }
    StructType *getType() { return Ty; }
  };

  class Nanos6MultidepFactory {
    const size_t MAX_DEP_DIMS = 8;
  private:
    StringMap<FunctionCallee> DepNameToFuncCalleeMap;

    FunctionType *BuildDepFuncType(Module &M, StringRef FullName, size_t Ndims, bool IsReduction) {
      // void nanos6_register_region_X_depinfoY(
      //   void *handler, int symbol_index, char const *region_text,
      //   void *base_address,
      //   long dim1size, long dim1start, long dim1end,
      //   ...);
      //
      // Except for reductions
      // void nanos6_register_region_reduction_depinfoY(
      //   int reduction_operation, int reduction_index,
      //   void *handler, int symbol_index, char const *region_text,
      //   void *base_address,
      //   long dim1size, long dim1start, long dim1end,
      //   ...);
      SmallVector<Type *, 8> Params;
      if (IsReduction) {
        Params.append({
          Type::getInt32Ty(M.getContext()),
          Type::getInt32Ty(M.getContext())
        });
      }
      Params.append({
        Type::getInt8PtrTy(M.getContext()),
        Type::getInt32Ty(M.getContext()),
        Type::getInt8PtrTy(M.getContext()),
        Type::getInt8PtrTy(M.getContext())
      });
      for (size_t i = 0; i < Ndims; ++i) {
        // long dimsize
        Params.push_back(Type::getInt64Ty(M.getContext()));
        // long dimstart
        Params.push_back(Type::getInt64Ty(M.getContext()));
        // long dimend
        Params.push_back(Type::getInt64Ty(M.getContext()));
      }
      return FunctionType::get(Type::getVoidTy(M.getContext()),
                               Params, /*IsVarArgs=*/false);
    }
  public:
    FunctionCallee getMultidepFuncCallee(Module &M, StringRef Name, size_t Ndims, bool IsReduction=false) {
      std::string FullName = ("nanos6_register_region_" + Name + "_depinfo" + Twine(Ndims)).str();

      auto It = DepNameToFuncCalleeMap.find(FullName);
      if (It != DepNameToFuncCalleeMap.end())
        return It->second;

      assert(Ndims <= MAX_DEP_DIMS);

      FunctionType *DepF = BuildDepFuncType(M, FullName, Ndims, IsReduction);
      FunctionCallee DepCallee = M.getOrInsertFunction(FullName, DepF);
      DepNameToFuncCalleeMap[FullName] = DepCallee;
      return DepCallee;
    }
  };
  Nanos6MultidepFactory MultidepFactory;

  FunctionCallee CreateTaskFuncCallee;
  FunctionCallee TaskSubmitFuncCallee;
  FunctionCallee RegisterLoopFuncCallee;
  FunctionCallee TaskInFinalFuncCallee;
  FunctionCallee TaskInfoRegisterFuncCallee;
  FunctionCallee TaskInfoRegisterCtorFuncCallee;

  // Data used to build final code.
  // We use Instructions instead of BasicBlocks because
  // BasicBlock pointers/iterators are invalidated after
  // splitBasicBlock
  struct FinalBodyInfo {
    Instruction *CloneEntry;
    Instruction *CloneExit;
    TaskInfo *TI;
  };


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

  void buildLoopForTaskImpl(Module &M, Function &F, Instruction *Entry,
                        Instruction *Exit, TaskLoopInfo &LoopInfo,
                        BasicBlock *&LoopEntryBB, BasicBlock *&BodyBB) {

    IRBuilder<> IRB(Entry);
    Type *IndVarTy = LoopInfo.IndVar->getType()->getPointerElementType();

    IRB.CreateStore(IRB.CreateSExtOrTrunc(LoopInfo.LBound, IndVarTy), LoopInfo.IndVar);

    BasicBlock *CondBB = IRB.saveIP().getBlock()->splitBasicBlock(IRB.saveIP().getPoint());
    CondBB->setName("for.cond");

    // The new entry is the start of the loop
    LoopEntryBB = CondBB->getUniquePredecessor();

    IRB.SetInsertPoint(Entry);

    Value *IndVarVal = IRB.CreateLoad(LoopInfo.IndVar);
    Value *LoopCmp = IRB.CreateSExtOrTrunc(LoopInfo.UBound, IndVarTy);
    switch (LoopInfo.LoopType) {
    case TaskLoopInfo::SLT:
      LoopCmp = IRB.CreateICmpSLT(IndVarVal, LoopCmp);
      break;
    case TaskLoopInfo::SLE:
      LoopCmp = IRB.CreateICmpSLE(IndVarVal, LoopCmp);
      break;
    case TaskLoopInfo::SGT:
      LoopCmp = IRB.CreateICmpSGT(IndVarVal, LoopCmp);
      break;
    case TaskLoopInfo::SGE:
      LoopCmp = IRB.CreateICmpSGE(IndVarVal, LoopCmp);
      break;
    case TaskLoopInfo::ULT:
      LoopCmp = IRB.CreateICmpULT(IndVarVal, LoopCmp);
      break;
    case TaskLoopInfo::ULE:
      LoopCmp = IRB.CreateICmpULE(IndVarVal, LoopCmp);
      break;
    case TaskLoopInfo::UGT:
      LoopCmp = IRB.CreateICmpUGT(IndVarVal, LoopCmp);
      break;
    case TaskLoopInfo::UGE:
      LoopCmp = IRB.CreateICmpUGE(IndVarVal, LoopCmp);
      break;
    default:
      llvm_unreachable("unexpected loop type");
    }
    BodyBB = IRB.saveIP().getBlock()->splitBasicBlock(IRB.saveIP().getPoint());
    BodyBB->setName("for.body");

    // Replace default br. by a conditional br. to task.body or task end
    Instruction *OldTerminator = CondBB->getTerminator();
    IRB.SetInsertPoint(OldTerminator);
    IRB.CreateCondBr(LoopCmp, BodyBB, Exit->getParent()->getUniqueSuccessor());
    OldTerminator->eraseFromParent();

    BasicBlock *IncrBB = BasicBlock::Create(M.getContext(), "for.incr", &F);

    // Add a br. to for.cond
    IRB.SetInsertPoint(IncrBB);
    IndVarVal = IRB.CreateLoad(LoopInfo.IndVar);
    IndVarVal = IRB.CreateAdd(IndVarVal, IRB.CreateSExtOrTrunc(LoopInfo.Step, IndVarTy));
    IRB.CreateStore(IndVarVal, LoopInfo.IndVar);
    IRB.CreateBr(CondBB);

    // Replace task end br. by a br. to for.incr
    OldTerminator = Exit->getParent()->getTerminator();
    assert(OldTerminator->getNumSuccessors() == 1);
    OldTerminator->setSuccessor(0, IncrBB);
  }

  BasicBlock *buildLoopForTask(
      Module &M, Function &F, Instruction *Entry,
      Instruction *Exit, TaskLoopInfo &LoopInfo) {
    BasicBlock *EntryBB = nullptr;
    BasicBlock *LoopEntryBB = nullptr;
    buildLoopForTaskImpl(M, F, Entry, Exit, LoopInfo, LoopEntryBB, EntryBB);
    return LoopEntryBB;
  }

  // Insert a new nanos6 task info registration in
  // the constructor (global ctor inserted) function
  void registerTaskInfo(Module &M, Value *TaskInfoVar) {
    Function *Func = cast<Function>(TaskInfoRegisterCtorFuncCallee.getCallee());
    BasicBlock &Entry = Func->getEntryBlock();

    IRBuilder<> BBBuilder(&Entry.getInstList().back());
    BBBuilder.CreateCall(TaskInfoRegisterFuncCallee, TaskInfoVar);
  }

  void unpackDestroyArgsAndRewrite(
      Module &M, const TaskInfo &TI, Function *F,
      const MapVector<Value *, size_t> &StructToIdxMap) {

    BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock &Entry = F->getEntryBlock();
    IRBuilder<> IRB(&Entry);

    for (Value *V : TI.DSAInfo.Private) {
      // Call custom destructor in clang in non-pods
      auto It = TI.NonPODsInfo.Deinits.find(V);
      if (It != TI.NonPODsInfo.Deinits.end()) {
        Type *Ty = V->getType()->getPointerElementType();
        // Compute num elements
        Value *NSize = ConstantInt::get(IRB.getInt64Ty(), 1);
        if (isa<ArrayType>(Ty)) {
          while (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
            // Constant array
            Value *NumElems = ConstantInt::get(IRB.getInt64Ty(),
                                               ArrTy->getNumElements());
            NSize = IRB.CreateNUWMul(NSize, NumElems);
            Ty = ArrTy->getElementType();
          }
        } else if (TI.VLADimsInfo.count(V)) {
          for (Value *Dim : TI.VLADimsInfo.lookup(V))
            NSize = IRB.CreateNUWMul(NSize, F->getArg(StructToIdxMap.lookup(Dim)));
        }

        // Regular arrays have types like [10 x %struct.S]*
        // Cast to %struct.S*
        Value *FArg = IRB.CreateBitCast(F->getArg(StructToIdxMap.lookup(V)), Ty->getPointerTo());

        llvm::Function *Func = cast<Function>(It->second);
        IRB.CreateCall(Func, ArrayRef<Value*>{FArg, NSize});
      }
    }
    for (Value *V : TI.DSAInfo.Firstprivate) {
      // Call custom destructor in clang in non-pods
      auto It = TI.NonPODsInfo.Deinits.find(V);
      if (It != TI.NonPODsInfo.Deinits.end()) {
        Type *Ty = V->getType()->getPointerElementType();
        // Compute num elements
        Value *NSize = ConstantInt::get(IRB.getInt64Ty(), 1);
        if (isa<ArrayType>(Ty)) {
          while (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
            // Constant array
            Value *NumElems = ConstantInt::get(IRB.getInt64Ty(),
                                               ArrTy->getNumElements());
            NSize = IRB.CreateNUWMul(NSize, NumElems);
            Ty = ArrTy->getElementType();
          }
        } else if (TI.VLADimsInfo.count(V)) {
          for (Value *Dim : TI.VLADimsInfo.lookup(V))
            NSize = IRB.CreateNUWMul(NSize, F->getArg(StructToIdxMap.lookup(Dim)));
        }

        // Regular arrays have types like [10 x %struct.S]*
        // Cast to %struct.S*
        Value *FArg = IRB.CreateBitCast(F->getArg(StructToIdxMap.lookup(V)), Ty->getPointerTo());

        llvm::Function *Func = cast<Function>(It->second);
        IRB.CreateCall(Func, ArrayRef<Value*>{FArg, NSize});
      }
    }
    IRB.CreateRetVoid();
  }

  void unpackDepsAndRewrite(Module &M, const TaskInfo &TI,
                            Function *F,
                            const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock &Entry = F->getEntryBlock();

    // Once we have inserted the cloned instructions and the ConstantExpr instructions
    // add the terminator so IRBuilder inserts just before it
    F->getEntryBlock().getInstList().push_back(ReturnInst::Create(M.getContext()));

    // Insert RT call before replacing uses
    unpackDepsCallToRT(M, TI, F);

    if (!TI.LoopInfo.empty()) {
      Type *IndVarTy = TI.LoopInfo.IndVar->getType()->getPointerElementType();

      IRBuilder<> IRB(&Entry.front());
      Value *LoopBounds = &*(F->arg_end() - 2);

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
      Value *LBoundField = IRB.CreateGEP(LoopBounds, Idx, "lb_gep");
      LBoundField = IRB.CreateLoad(LBoundField);
      LBoundField = IRB.CreateSExtOrTrunc(LBoundField, IndVarTy, "lb");

      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 1);
      Value *UBoundField = IRB.CreateGEP(LoopBounds, Idx, "ub_gep");
      UBoundField = IRB.CreateLoad(UBoundField);
      UBoundField = IRB.CreateSExtOrTrunc(UBoundField, IndVarTy);
      UBoundField = IRB.CreateSub(UBoundField, ConstantInt::get(IndVarTy, 1), "ub");

      // Replace loop bounds
      for (Instruction &I : Entry) {
        if (isReplaceableValue(TI.LoopInfo.LBound))
          I.replaceUsesOfWith(TI.LoopInfo.LBound, LBoundField);
        if (isReplaceableValue(TI.LoopInfo.UBound))
        I.replaceUsesOfWith(TI.LoopInfo.UBound, UBoundField);
      }
    }

    for (Instruction &I : Entry) {
      Function::arg_iterator AI = F->arg_begin();
      for (auto It = StructToIdxMap.begin();
             It != StructToIdxMap.end(); ++It, ++AI) {
        if (isReplaceableValue(It->first))
          I.replaceUsesOfWith(It->first, &*AI);
      }
    }
  }

  void unpackCostAndRewrite(Module &M, Value *Cost, Function *F,
                            const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock &Entry = F->getEntryBlock();
    F->getEntryBlock().getInstList().push_back(ReturnInst::Create(M.getContext()));
    IRBuilder<> BBBuilder(&F->getEntryBlock().back());
    Value *Constraints = &*(F->arg_end() - 1);
    Value *Idx[2];
    Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
    Idx[1] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));

    Value *GEPConstraints = BBBuilder.CreateGEP(
          Constraints, Idx, "gep_" + Constraints->getName());
    Value *CostCast = BBBuilder.CreateZExt(Cost, Nanos6TaskConstraints::getInstance(M).getType()->getElementType(0));
    BBBuilder.CreateStore(CostCast, GEPConstraints);
    for (Instruction &I : Entry) {
      Function::arg_iterator AI = F->arg_begin();
      for (auto It = StructToIdxMap.begin();
             It != StructToIdxMap.end(); ++It, ++AI) {
        if (isReplaceableValue(It->first))
          I.replaceUsesOfWith(It->first, &*AI);
      }
    }
  }

  void unpackPriorityAndRewrite(Module &M, Value *Priority, Function *F,
                                const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock &Entry = F->getEntryBlock();
    F->getEntryBlock().getInstList().push_back(ReturnInst::Create(M.getContext()));
    IRBuilder<> BBBuilder(&F->getEntryBlock().back());
    Value *PriorityArg = &*(F->arg_end() - 1);
    Value *PrioritySExt = BBBuilder.CreateSExt(Priority, Type::getInt64Ty(M.getContext()));
    BBBuilder.CreateStore(PrioritySExt, PriorityArg);
    for (Instruction &I : Entry) {
      Function::arg_iterator AI = F->arg_begin();
      for (auto It = StructToIdxMap.begin();
             It != StructToIdxMap.end(); ++It, ++AI) {
        if (isReplaceableValue(It->first))
          I.replaceUsesOfWith(It->first, &*AI);
      }
    }
  }

  void unpackCallToRTOfType(Module &M,
                            const SmallVectorImpl<DependInfo> &DependList,
                            Function *F,
                            StringRef DepType) {
    for (const DependInfo &DI : DependList) {
      BasicBlock &Entry = F->getEntryBlock();
      Instruction &RetI = Entry.back();
      IRBuilder<> BBBuilder(&RetI);

      Function *ComputeDepFun = cast<Function>(DI.ComputeDepFun);
      CallInst *CallComputeDep = BBBuilder.CreateCall(ComputeDepFun, DI.Args);
      StructType *ComputeDepTy = cast<StructType>(ComputeDepFun->getReturnType());

      assert(ComputeDepTy->getNumElements() > 1 && "Expected dependency base with dim_{size, start, end}");
      size_t NumDims = (ComputeDepTy->getNumElements() - 1)/3;

      llvm::Value *Base = BBBuilder.CreateExtractValue(CallComputeDep, 0);

      SmallVector<Value *, 4> TaskDepAPICall;
      Value *Handler = &*(F->arg_end() - 1);
      TaskDepAPICall.push_back(Handler);
      TaskDepAPICall.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()), DI.SymbolIndex));
      TaskDepAPICall.push_back(ConstantPointerNull::get(Type::getInt8PtrTy(M.getContext()))); // TODO: stringify
      TaskDepAPICall.push_back(BBBuilder.CreateBitCast(Base, Type::getInt8PtrTy(M.getContext())));
      for (size_t i = 1; i < ComputeDepTy->getNumElements(); ++i) {
        TaskDepAPICall.push_back(BBBuilder.CreateExtractValue(CallComputeDep, i));
      }

      BBBuilder.CreateCall(MultidepFactory.getMultidepFuncCallee(M, DepType, NumDims), TaskDepAPICall);
    }
  }

  void unpackCallToRTOfReduction(Module &M,
                            const SmallVectorImpl<ReductionInfo> &ReductionsList,
                            const TaskReductionsInitCombInfo &TRI,
                            Function *F,
                            StringRef RedType) {
    for (const ReductionInfo &RI : ReductionsList) {
      const DependInfo &DI = RI.DepInfo;
      BasicBlock &Entry = F->getEntryBlock();
      Instruction &RetI = Entry.back();
      IRBuilder<> BBBuilder(&RetI);

      // Do remove ComputeDep, we're going to use it in ol_task_region
      Function *ComputeDepFun = cast<Function>(DI.ComputeDepFun);
      CallInst *CallComputeDep = BBBuilder.CreateCall(ComputeDepFun, DI.Args);
      StructType *ComputeDepTy = cast<StructType>(ComputeDepFun->getReturnType());

      llvm::Value *DepBaseDSA = DI.Args[0];
      // This must not happen, it will be catched in analysis
      assert(TRI.count(DepBaseDSA) && "Reduction dependency DSA has no init/combiner");

      assert(ComputeDepTy->getNumElements() > 1 && "Expected dependency base with dim_{size, start, end}");
      size_t NumDims = (ComputeDepTy->getNumElements() - 1)/3;

      llvm::Value *Base = BBBuilder.CreateExtractValue(CallComputeDep, 0);

      SmallVector<Value *, 4> TaskDepAPICall;
      TaskDepAPICall.push_back(RI.RedKind);
      TaskDepAPICall.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()), TRI.lookup(Base).ReductionIndex));
      Value *Handler = &*(F->arg_end() - 1);
      TaskDepAPICall.push_back(Handler);
      TaskDepAPICall.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()), DI.SymbolIndex));
      TaskDepAPICall.push_back(ConstantPointerNull::get(Type::getInt8PtrTy(M.getContext()))); // TODO: stringify
      TaskDepAPICall.push_back(BBBuilder.CreateBitCast(Base, Type::getInt8PtrTy(M.getContext())));
      for (size_t i = 1; i < ComputeDepTy->getNumElements(); ++i) {
        TaskDepAPICall.push_back(BBBuilder.CreateExtractValue(CallComputeDep, i));
      }
      BBBuilder.CreateCall(MultidepFactory.getMultidepFuncCallee(M, RedType, NumDims, /*IsReduction=*/true), TaskDepAPICall);
    }
  }

  void unpackDepsCallToRT(Module &M,
                      const TaskInfo &TI,
                      Function *F) {
    const TaskDependsInfo &TDI = TI.DependsInfo;
    const TaskReductionsInitCombInfo &TRI = TI.ReductionsInitCombInfo;

    unpackCallToRTOfType(M, TDI.Ins, F, "read");
    unpackCallToRTOfType(M, TDI.Outs, F, "write");
    unpackCallToRTOfType(M, TDI.Inouts, F, "readwrite");
    unpackCallToRTOfType(M, TDI.Concurrents, F, "concurrent");
    unpackCallToRTOfType(M, TDI.Commutatives, F, "commutative");
    unpackCallToRTOfType(M, TDI.WeakIns, F, "weak_read");
    unpackCallToRTOfType(M, TDI.WeakOuts, F, "weak_write");
    unpackCallToRTOfType(M, TDI.WeakInouts, F, "weak_readwrite");
    unpackCallToRTOfType(M, TDI.WeakConcurrents, F, "weak_concurrent");
    unpackCallToRTOfType(M, TDI.WeakCommutatives, F, "weak_commutative");
    unpackCallToRTOfReduction(M, TDI.Reductions, TRI, F, "reduction");
    unpackCallToRTOfReduction(M, TDI.WeakReductions, TRI, F, "weak_reduction");
  }

  // TypeList[i] <-> NameList[i]
  // ExtraTypeList[i] <-> ExtraNameList[i]
  Function *createUnpackOlFunction(Module &M, Function &F,
                                 std::string Name,
                                 ArrayRef<Type *> TypeList,
                                 ArrayRef<StringRef> NameList,
                                 ArrayRef<Type *> ExtraTypeList,
                                 ArrayRef<StringRef> ExtraNameList) {
    Type *RetTy = Type::getVoidTy(M.getContext());

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

    // Set names for arguments.
    Function::arg_iterator AI = FuncVar->arg_begin();
    for (unsigned i = 0, e = AggNameList.size(); i != e; ++i, ++AI)
      AI->setName(AggNameList[i]);

    return FuncVar;
  }

  // Build a new storage for the translated reduction
  // returns the storage of the translated reduction
  void translateReductionUnpackedDSA(IRBuilder<> &IRB, const DependInfo &DI,
                                     Value *DSA, Value *&UnpackedDSA,
                                     Value *AddrTranslationTable,
                                     const std::map<Value *, int> &DepSymToIdx) {
    Function *ComputeDepFun = cast<Function>(DI.ComputeDepFun);
    CallInst *CallComputeDep = IRB.CreateCall(ComputeDepFun, DI.Args);
    llvm::Value *Base = IRB.CreateExtractValue(CallComputeDep, 0);

    // Save the original type since we are going to cast...
    Type *UnpackedDSAType = UnpackedDSA->getType();
    Type *BaseType = Base->getType();

    // Storage of the translated DSA
    AllocaInst *UnpackedDSATranslated = IRB.CreateAlloca(BaseType);

    Value *Idx[2];
    Idx[0] = ConstantInt::get(Type::getInt32Ty(IRB.getContext()), DepSymToIdx.at(DSA));
    Idx[1] = Constant::getNullValue(Type::getInt32Ty(IRB.getContext()));
    Value *LocalAddr = IRB.CreateGEP(
        AddrTranslationTable, Idx, "local_lookup_" + DSA->getName());
    LocalAddr = IRB.CreateLoad(LocalAddr);

    Idx[1] = ConstantInt::get(Type::getInt32Ty(IRB.getContext()), 1);
    Value *DeviceAddr = IRB.CreateGEP(
        AddrTranslationTable, Idx, "device_lookup_" + DSA->getName());
    DeviceAddr = IRB.CreateLoad(DeviceAddr);

    // Res = device_addr + (DSA_addr - local_addr)
    Base = IRB.CreateBitCast(Base, Type::getInt8PtrTy(IRB.getContext()));
    UnpackedDSA = IRB.CreateGEP(Base, IRB.CreateNeg(LocalAddr));
    UnpackedDSA = IRB.CreateGEP(UnpackedDSA, DeviceAddr);
    UnpackedDSA = IRB.CreateBitCast(UnpackedDSA, BaseType );

    IRB.CreateStore(UnpackedDSA, UnpackedDSATranslated);

   // FIXME: Since we have no info about if we have to pass to unpack a load of the alloca
   // or not, check if the type has changed after call to compute_dep.
   // Pointers -> no load
   // basic types/structs/arrays/vla -> load
   if (UnpackedDSAType == BaseType)
      UnpackedDSA = IRB.CreateLoad(UnpackedDSATranslated);
   else
      UnpackedDSA = UnpackedDSATranslated;
  }

  // Given a Outline Function assuming that task args are the first parameter, and
  // DSAInfo and VLADimsInfo, it unpacks task args in Outline and fills UnpackedList
  // with those Values, used to call Unpack Functions
  void unpackDSAsWithVLADims(Module &M, const TaskInfo &TI,
                  Function *OlFunc,
                  const MapVector<Value *, size_t> &StructToIdxMap,
                  SmallVectorImpl<Value *> &UnpackedList) {
    UnpackedList.clear();

    const TaskDSAInfo &DSAInfo = TI.DSAInfo;
    const TaskCapturedInfo &CapturedInfo = TI.CapturedInfo;
    const TaskVLADimsInfo &VLADimsInfo = TI.VLADimsInfo;

    IRBuilder<> BBBuilder(&OlFunc->getEntryBlock());
    Function::arg_iterator AI = OlFunc->arg_begin();
    Value *OlDepsFuncTaskArgs = &*AI++;
    for (Value *V : DSAInfo.Shared) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), StructToIdxMap.lookup(V));
      Value *GEP = BBBuilder.CreateGEP(
          OlDepsFuncTaskArgs, Idx, "gep_" + V->getName());
      Value *LGEP = BBBuilder.CreateLoad(GEP, "load_" + GEP->getName());

      UnpackedList.push_back(LGEP);
    }
    for (Value *V : DSAInfo.Private) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), StructToIdxMap.lookup(V));
      Value *GEP = BBBuilder.CreateGEP(
          OlDepsFuncTaskArgs, Idx, "gep_" + V->getName());

      // VLAs
      if (VLADimsInfo.count(V))
        GEP = BBBuilder.CreateLoad(GEP, "load_" + GEP->getName());

      UnpackedList.push_back(GEP);
    }
    for (Value *V : DSAInfo.Firstprivate) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), StructToIdxMap.lookup(V));
      Value *GEP = BBBuilder.CreateGEP(
          OlDepsFuncTaskArgs, Idx, "gep_" + V->getName());

      // VLAs
      if (VLADimsInfo.count(V))
        GEP = BBBuilder.CreateLoad(GEP, "load_" + GEP->getName());

      UnpackedList.push_back(GEP);
    }
    for (Value *V : CapturedInfo) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), StructToIdxMap.lookup(V));
      Value *GEP = BBBuilder.CreateGEP(
          OlDepsFuncTaskArgs, Idx, "capt_gep" + V->getName());
      Value *LGEP = BBBuilder.CreateLoad(GEP, "load_" + GEP->getName());
      UnpackedList.push_back(LGEP);
    }
  }

  // Given an Outline and Unpack Functions it unpacks DSAs in Outline
  // and builds a call to Unpack
  void olCallToUnpack(Module &M, const TaskInfo &TI,
                      MapVector<Value *, size_t> &StructToIdxMap,
                      Function *OlFunc, Function *UnpackFunc,
                      bool IsTaskFunc=false) {
    BasicBlock::Create(M.getContext(), "entry", OlFunc);
    IRBuilder<> BBBuilder(&OlFunc->getEntryBlock());

    // First arg is the nanos_task_args
    Function::arg_iterator AI = OlFunc->arg_begin();
    AI++;
    SmallVector<Value *, 4> UnpackParams;
    unpackDSAsWithVLADims(M, TI, OlFunc, StructToIdxMap, UnpackParams);

    if (IsTaskFunc) {
      // Build call to compute_dep in order to have get the base dependency of
      // the reduction. The result is passed to unpack
      ArrayRef<ReductionInfo> Reds = TI.DependsInfo.Reductions;
      ArrayRef<ReductionInfo> WeakReds = TI.DependsInfo.WeakReductions;
      // NOTE: this assumes UnpackParams can be indexed with StructToIdxMap
      Value *AddrTranslationTable = &*(OlFunc->arg_end() - 1);
      // Preserve the params before translation. And replace used after build all
      // compute_dep calls
      SmallVector<Value *, 4> UnpackParamsCopy(UnpackParams);
      for (auto &RedInfo : Reds) {
        Value *DepBaseDSA = RedInfo.DepInfo.Args[0];
        translateReductionUnpackedDSA(BBBuilder, RedInfo.DepInfo, DepBaseDSA,
                                      UnpackParams[StructToIdxMap[DepBaseDSA]],
                                      AddrTranslationTable, TI.DSAInfo.DepSymToIdx);
      }
      for (auto &RedInfo : WeakReds) {
        Value *DepBaseDSA = RedInfo.DepInfo.Args[0];
        translateReductionUnpackedDSA(BBBuilder, RedInfo.DepInfo, DepBaseDSA,
                                      UnpackParams[StructToIdxMap[DepBaseDSA]],
                                      AddrTranslationTable, TI.DSAInfo.DepSymToIdx);
      }
      for (Instruction &I : *BBBuilder.GetInsertBlock()) {
        auto UnpackedIt = UnpackParamsCopy.begin();
        for (auto It = StructToIdxMap.begin();
               It != StructToIdxMap.end(); ++It, ++UnpackedIt) {
          if (isReplaceableValue(It->first))
            I.replaceUsesOfWith(It->first, *UnpackedIt);
        }
      }
    }

    while (AI != OlFunc->arg_end()) {
      UnpackParams.push_back(&*AI++);
    }
    // Build TaskUnpackCall
    BBBuilder.CreateCall(UnpackFunc, UnpackParams);
    // Make BB legal with a terminator to task outline function
    BBBuilder.CreateRetVoid();
  }

  // Copy task_args from src to dst, calling copyctors or ctors if
  // nonpods
  void duplicateArgs(Module &M, const TaskInfo &TI,
                     MapVector<Value *, size_t> &StructToIdxMap,
                     Function *OlFunc, StructType *TaskArgsTy) {
    BasicBlock::Create(M.getContext(), "entry", OlFunc);
    IRBuilder<> IRB(&OlFunc->getEntryBlock());

    Function::arg_iterator AI = OlFunc->arg_begin();
    Value *TaskArgsSrc = &*AI++;
    Value *TaskArgsDst = &*AI++;
    Value *TaskArgsDstL = IRB.CreateLoad(TaskArgsDst);

    SmallVector<VLAAlign, 2> VLAAlignsInfo;
    computeVLAsAlignOrder(M, VLAAlignsInfo, TI.VLADimsInfo);

    Value *TaskArgsStructSizeOf = ConstantInt::get(IRB.getInt64Ty(), M.getDataLayout().getTypeAllocSize(TaskArgsTy));

    Value *TaskArgsDstLi8 = IRB.CreateBitCast(TaskArgsDstL, IRB.getInt8PtrTy());
    Value *TaskArgsDstLi8IdxGEP = IRB.CreateGEP(TaskArgsDstLi8, TaskArgsStructSizeOf, "args_end");

    // First point VLAs to its according space in task args
    for (const VLAAlign& VAlign : VLAAlignsInfo) {
      Value *const V = VAlign.V;
      unsigned TyAlign = VAlign.Align;

      Type *Ty = V->getType()->getPointerElementType();

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
      Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap[V]);
      Value *GEP = IRB.CreateGEP(
          TaskArgsDstL, Idx, "gep_dst_" + V->getName());

      // Point VLA in task args to an aligned position of the extra space allocated
      Value *GEPi8 = IRB.CreateBitCast(GEP, IRB.getInt8PtrTy()->getPointerTo());
      IRB.CreateAlignedStore(TaskArgsDstLi8IdxGEP, GEPi8, Align(TyAlign));
      // Skip current VLA size
      unsigned SizeB = M.getDataLayout().getTypeAllocSize(Ty);
      Value *VLASize = ConstantInt::get(IRB.getInt64Ty(), SizeB);
      for (Value *const &Dim : TI.VLADimsInfo.lookup(V)) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(Dim));
        Value *GEPDst = IRB.CreateGEP(
          TaskArgsDstL, Idx, "gep_dst_" + Dim->getName());
        GEPDst = IRB.CreateLoad(GEPDst);
        VLASize = IRB.CreateNUWMul(VLASize, GEPDst);
      }
      TaskArgsDstLi8IdxGEP = IRB.CreateGEP(TaskArgsDstLi8IdxGEP, VLASize);
    }

    for (Value *V : TI.DSAInfo.Shared) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
      Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap[V]);
      Value *GEPSrc = IRB.CreateGEP(
          TaskArgsSrc, Idx, "gep_src_" + V->getName());
      Value *GEPDst = IRB.CreateGEP(
          TaskArgsDstL, Idx, "gep_dst_" + V->getName());
      IRB.CreateStore(IRB.CreateLoad(GEPSrc), GEPDst);
    }
    for (Value *V : TI.DSAInfo.Private) {
      // Call custom constructor generated in clang in non-pods
      // Leave pods unititialized
      auto It = TI.NonPODsInfo.Inits.find(V);
      if (It != TI.NonPODsInfo.Inits.end()) {
        Type *Ty = V->getType()->getPointerElementType();
        // Compute num elements
        Value *NSize = ConstantInt::get(IRB.getInt64Ty(), 1);
        if (isa<ArrayType>(Ty)) {
          while (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
            // Constant array
            Value *NumElems = ConstantInt::get(IRB.getInt64Ty(),
                                               ArrTy->getNumElements());
            NSize = IRB.CreateNUWMul(NSize, NumElems);
            Ty = ArrTy->getElementType();
          }
        } else if (TI.VLADimsInfo.count(V)) {
          for (Value *const &Dim : TI.VLADimsInfo.lookup(V)) {
            Value *Idx[2];
            Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
            Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(Dim));
            Value *GEPSrc = IRB.CreateGEP(
              TaskArgsSrc, Idx, "gep_src_" + Dim->getName());
            GEPSrc = IRB.CreateLoad(GEPSrc);
            NSize = IRB.CreateNUWMul(NSize, GEPSrc);
          }
        }

        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap[V]);
        Value *GEP = IRB.CreateGEP(
            TaskArgsDstL, Idx, "gep_" + V->getName());

        // VLAs
        if (TI.VLADimsInfo.count(V))
          GEP = IRB.CreateLoad(GEP);

        // Regular arrays have types like [10 x %struct.S]*
        // Cast to %struct.S*
        GEP = IRB.CreateBitCast(GEP, Ty->getPointerTo());

        IRB.CreateCall(FunctionCallee(cast<Function>(It->second)), ArrayRef<Value*>{GEP, NSize});
      }
    }
    for (Value *V : TI.DSAInfo.Firstprivate) {
      Type *Ty = V->getType()->getPointerElementType();
      unsigned TyAlign = M.getDataLayout().getPrefTypeAlignment(Ty);

      // Compute num elements
      Value *NSize = ConstantInt::get(IRB.getInt64Ty(), 1);
      if (isa<ArrayType>(Ty)) {
        while (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
          // Constant array
          Value *NumElems = ConstantInt::get(IRB.getInt64Ty(),
                                             ArrTy->getNumElements());
          NSize = IRB.CreateNUWMul(NSize, NumElems);
          Ty = ArrTy->getElementType();
        }
      } else if (TI.VLADimsInfo.count(V)) {
        for (Value *const &Dim : TI.VLADimsInfo.lookup(V)) {
          Value *Idx[2];
          Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
          Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(Dim));
          Value *GEPSrc = IRB.CreateGEP(
            TaskArgsSrc, Idx, "gep_src_" + Dim->getName());
          GEPSrc = IRB.CreateLoad(GEPSrc);
          NSize = IRB.CreateNUWMul(NSize, GEPSrc);
        }
      }

      // call custom copy constructor generated in clang in non-pods
      // do a memcpy if pod
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
      Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap[V]);
      Value *GEPSrc = IRB.CreateGEP(
          TaskArgsSrc, Idx, "gep_src_" + V->getName());
      Value *GEPDst = IRB.CreateGEP(
          TaskArgsDstL, Idx, "gep_dst_" + V->getName());

      // VLAs
      if (TI.VLADimsInfo.count(V)) {
        GEPSrc = IRB.CreateLoad(GEPSrc);
        GEPDst = IRB.CreateLoad(GEPDst);
      }

      auto It = TI.NonPODsInfo.Copies.find(V);
      if (It != TI.NonPODsInfo.Copies.end()) {
        // Non-POD

        // Regular arrays have types like [10 x %struct.S]*
        // Cast to %struct.S*
        GEPSrc = IRB.CreateBitCast(GEPSrc, Ty->getPointerTo());
        GEPDst = IRB.CreateBitCast(GEPDst, Ty->getPointerTo());


        llvm::Function *Func = cast<Function>(It->second);
        IRB.CreateCall(Func, ArrayRef<Value*>{/*Src=*/GEPSrc, /*Dst=*/GEPDst, NSize});
      } else {
        unsigned SizeB = M.getDataLayout().getTypeAllocSize(Ty);
        Value *NSizeB = IRB.CreateNUWMul(NSize, ConstantInt::get(IRB.getInt64Ty(), SizeB));
        IRB.CreateMemCpy(GEPDst, Align(TyAlign), GEPSrc, Align(TyAlign), NSizeB);
      }
    }
    for (Value *V : TI.CapturedInfo) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
      Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap[V]);
      Value *GEPSrc = IRB.CreateGEP(
          TaskArgsSrc, Idx, "capt_gep_src_" + V->getName());
      Value *GEPDst = IRB.CreateGEP(
          TaskArgsDstL, Idx, "capt_gep_dst_" + V->getName());
      IRB.CreateStore(IRB.CreateLoad(GEPSrc), GEPDst);
    }

    IRB.CreateRetVoid();
  }

  Value *computeTaskArgsVLAsExtraSizeOf(Module &M, IRBuilder<> &IRB, const TaskVLADimsInfo &VLADimsInfo) {
    Value *Sum = ConstantInt::get(IRB.getInt64Ty(), 0);
    for (auto &VLAWithDimsMap : VLADimsInfo) {
      Type *Ty = VLAWithDimsMap.first->getType()->getPointerElementType();
      unsigned SizeB = M.getDataLayout().getTypeAllocSize(Ty);
      Value *ArraySize = ConstantInt::get(IRB.getInt64Ty(), SizeB);
      for (Value *const &V : VLAWithDimsMap.second) {
        ArraySize = IRB.CreateNUWMul(ArraySize, V);
      }
      Sum = IRB.CreateNUWAdd(Sum, ArraySize);
    }
    return Sum;
  }

  StructType *createTaskArgsType(Module &M,
                                 const TaskInfo &TI,
                                 MapVector<Value *, size_t> &StructToIdxMap, StringRef Str) {
    const TaskDSAInfo &DSAInfo = TI.DSAInfo;
    const TaskCapturedInfo &CapturedInfo = TI.CapturedInfo;
    const TaskVLADimsInfo &VLADimsInfo = TI.VLADimsInfo;
    // Private and Firstprivate must be stored in the struct
    // Captured values (i.e. VLA dimensions) are not pointers
    SmallVector<Type *, 4> TaskArgsMemberTy;
    size_t TaskArgsIdx = 0;
    for (Value *V : DSAInfo.Shared) {
      TaskArgsMemberTy.push_back(V->getType());
      StructToIdxMap[V] = TaskArgsIdx++;
    }
    for (Value *V : DSAInfo.Private) {
      // VLAs
      if (VLADimsInfo.count(V))
        TaskArgsMemberTy.push_back(V->getType());
      else
        TaskArgsMemberTy.push_back(V->getType()->getPointerElementType());
      StructToIdxMap[V] = TaskArgsIdx++;
    }
    for (Value *V : DSAInfo.Firstprivate) {
      // VLAs
      if (VLADimsInfo.count(V))
        TaskArgsMemberTy.push_back(V->getType());
      else
        TaskArgsMemberTy.push_back(V->getType()->getPointerElementType());
      StructToIdxMap[V] = TaskArgsIdx++;
    }
    for (Value *V : CapturedInfo) {
      assert(!V->getType()->isPointerTy() && "Captures are not pointers");
      TaskArgsMemberTy.push_back(V->getType());
      StructToIdxMap[V] = TaskArgsIdx++;
    }
    return StructType::create(M.getContext(), TaskArgsMemberTy, Str);
  }

  struct VLAAlign {
    Value *V;
    unsigned Align;
  };

  // Greater alignemt go first
  void computeVLAsAlignOrder(Module &M, SmallVectorImpl<VLAAlign> &VLAAlignsInfo, const TaskVLADimsInfo &VLADimsInfo) {
    for (const auto &VLAWithDimsMap : VLADimsInfo) {
      Value *const V = VLAWithDimsMap.first;
      Type *Ty = V->getType()->getPointerElementType();

      unsigned Align = M.getDataLayout().getPrefTypeAlignment(Ty);

      auto It = VLAAlignsInfo.begin();
      while (It != VLAAlignsInfo.end() && It->Align >= Align)
        ++It;

      VLAAlignsInfo.insert(It, {V, Align});
    }
  }

  void lowerTaskwait(const TaskwaitInfo &TwI,
                     Module &M) {
    // 1. Create Taskwait function Type
    IRBuilder<> IRB(TwI.I);
    FunctionCallee Func = M.getOrInsertFunction(
        "nanos6_taskwait", IRB.getVoidTy(), IRB.getInt8PtrTy());
    // 2. Build String
    unsigned Line = TwI.I->getDebugLoc().getLine();
    unsigned Col = TwI.I->getDebugLoc().getCol();

    std::string FileNamePlusLoc = (M.getSourceFileName()
                                   + ":" + Twine(Line)
                                   + ":" + Twine(Col)).str();
    Constant *Nanos6TaskwaitLocStr = IRB.CreateGlobalStringPtr(FileNamePlusLoc);

    // 3. Insert the call
    IRB.CreateCall(Func, {Nanos6TaskwaitLocStr});
    // 4. Remove the intrinsic
    TwI.I->eraseFromParent();
  }

  // This must be called before erasing original entry/exit
  void buildFinalCondCFG(
      Module &M, Function &F, ArrayRef<FinalBodyInfo> FinalInfos) {
    // Lower final inner tasks

    // Skip first since its the current task
    for (size_t i = 1; i < FinalInfos.size(); ++i) {
      // Build loop for taskloop/taskfor
      if (!FinalInfos[i].TI->LoopInfo.empty()) {
        buildLoopForTask(M, F, FinalInfos[i].CloneEntry, FinalInfos[i].CloneExit, FinalInfos[i].TI->LoopInfo);
      }
      FinalInfos[i].CloneExit->eraseFromParent();
      FinalInfos[i].CloneEntry->eraseFromParent();
    }

    BasicBlock *OrigEntryBB = FinalInfos[0].TI->Entry->getParent();
    BasicBlock *OrigExitBB = FinalInfos[0].TI->Exit->getParent()->getUniqueSuccessor();

    BasicBlock *CloneEntryBB = FinalInfos[0].CloneEntry->getParent();
    if (!FinalInfos[0].TI->LoopInfo.empty()) {
      CloneEntryBB = buildLoopForTask(M, F, FinalInfos[0].CloneEntry, FinalInfos[0].CloneExit, FinalInfos[0].TI->LoopInfo);
    }

    FinalInfos[0].CloneExit->eraseFromParent();
    FinalInfos[0].CloneEntry->eraseFromParent();

    OrigExitBB->setName("final.end");
    assert(OrigEntryBB->getSinglePredecessor());
    BasicBlock *FinalCondBB = BasicBlock::Create(M.getContext(), "final.cond", &F);

    BasicBlock *CopyEntryBB = CloneEntryBB;
    CopyEntryBB->setName("final.then");

    // We are now just before the branch to task body
    Instruction *EntryBBTerminator = OrigEntryBB->getSinglePredecessor()->getTerminator();

    IRBuilder<> IRB(EntryBBTerminator);

    IRB.CreateBr(FinalCondBB);
    // Remove the old branch
    EntryBBTerminator->eraseFromParent();

    IRB.SetInsertPoint(FinalCondBB);
    // if (nanos6_in_final())
    Value *Cond = IRB.CreateICmpNE(IRB.CreateCall(TaskInFinalFuncCallee, {}), IRB.getInt32(0));
    IRB.CreateCondBr(Cond, CopyEntryBB, OrigEntryBB);

  }

  void lowerTask(TaskInfo &TI,
                 Function &F,
                 size_t taskNum,
                 Module &M,
                 DenseMap<TaskInfo *, SmallVector<FinalBodyInfo, 4>> &TaskFinalInfo) {

    DebugLoc DLoc = TI.Entry->getDebugLoc();
    unsigned Line = DLoc.getLine();
    unsigned Col = DLoc.getCol();
    std::string FileNamePlusLoc = (M.getSourceFileName()
                                   + ":" + Twine(Line)
                                   + ":" + Twine(Col)).str();

    Constant *Nanos6TaskLocStr = IRBuilder<>(TI.Entry).CreateGlobalStringPtr(FileNamePlusLoc);

    buildFinalCondCFG(M, F, TaskFinalInfo[&TI]);

    BasicBlock *EntryBB = TI.Entry->getParent();
    BasicBlock *ExitBB = TI.Exit->getParent()->getUniqueSuccessor();

    // In loop constructs this will be the starting loop BB
    BasicBlock *NewEntryBB = EntryBB;
    TaskLoopInfo NewLoopInfo = TI.LoopInfo;
    if (!TI.LoopInfo.empty()) {
      Type *IndVarTy = TI.LoopInfo.IndVar->getType()->getPointerElementType();

      IRBuilder<> IRB(TI.Entry);

      // Use tmp variables to be replaced by what comes from nanos6. This fixes
      // the problem when bounds or step are constants
      NewLoopInfo.LBound = IRB.CreateAlloca(IndVarTy, nullptr, "lb.tmp.addr");
      IRB.CreateStore(TI.LoopInfo.LBound, NewLoopInfo.LBound);
      NewLoopInfo.LBound = IRB.CreateLoad(NewLoopInfo.LBound);

      NewLoopInfo.UBound = IRB.CreateAlloca(IndVarTy, nullptr, "ub.tmp.addr");
      IRB.CreateStore(TI.LoopInfo.UBound, NewLoopInfo.UBound);
      NewLoopInfo.UBound = IRB.CreateLoad(NewLoopInfo.UBound);

      // unpacked_task_region loops are always step 1
      NewLoopInfo.Step = IRB.CreateAlloca(IndVarTy, nullptr, "step.tmp.addr");
      IRB.CreateStore(ConstantInt::get(IndVarTy, 1), NewLoopInfo.Step);
      NewLoopInfo.Step = IRB.CreateLoad(NewLoopInfo.Step);

      NewLoopInfo.IndVar =
        IRB.CreateAlloca(IndVarTy, nullptr, "loop." + TI.LoopInfo.IndVar->getName());
      // unpacked_task_region loops are always SLT
      NewLoopInfo.LoopType = TaskLoopInfo::SLT;
      buildLoopForTaskImpl(M, F, TI.Entry, TI.Exit, NewLoopInfo, NewEntryBB, EntryBB);
    }

    TI.Exit->eraseFromParent();
    TI.Entry->eraseFromParent();

    SetVector<BasicBlock *> TaskBBs;
    SmallVector<BasicBlock*, 8> Worklist;
    SmallPtrSet<BasicBlock*, 8> Visited;

    // 2. Gather BB between entry and exit (is there any function/util to do this?)
    Worklist.push_back(NewEntryBB);
    Visited.insert(NewEntryBB);
    TaskBBs.insert(NewEntryBB);
    while (!Worklist.empty()) {
      auto WIt = Worklist.begin();
      BasicBlock *BB = *WIt;
      Worklist.erase(WIt);

      for (auto It = succ_begin(BB); It != succ_end(BB); ++It) {
        if (!Visited.count(*It) && *It != ExitBB) {
          Worklist.push_back(*It);
          Visited.insert(*It);
          TaskBBs.insert(*It);
        }
      }
    }

    // Create nanos6_task_args_* START
    SmallVector<Type *, 4> TaskArgsMemberTy;
    MapVector<Value *, size_t> TaskArgsToStructIdxMap;
    StructType *TaskArgsTy = createTaskArgsType(M, TI, TaskArgsToStructIdxMap,
                                                 ("nanos6_task_args_" + F.getName() + Twine(taskNum)).str());
    // Create nanos6_task_args_* END

    SmallVector<Type *, 4> TaskTypeList;
    SmallVector<StringRef, 4> TaskNameList;
    for (auto It = TaskArgsToStructIdxMap.begin();
           It != TaskArgsToStructIdxMap.end(); ++It) {
      Value *V = It->first;
      TaskTypeList.push_back(V->getType());
      TaskNameList.push_back(V->getName());
    }

    SmallVector<Type *, 4> TaskExtraTypeList;
    SmallVector<StringRef, 4> TaskExtraNameList;

    // nanos6_unpacked_destroy_* START
    TaskExtraTypeList.clear();
    TaskExtraNameList.clear();

    Function *UnpackDestroyArgsFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_unpacked_destroy_" + F.getName() + Twine(taskNum)).str(),
                               TaskTypeList, TaskNameList,
                               TaskExtraTypeList, TaskExtraNameList);
    unpackDestroyArgsAndRewrite(M, TI, UnpackDestroyArgsFuncVar, TaskArgsToStructIdxMap);

    // nanos6_unpacked_destroy_* END

    // nanos6_ol_destroy_* START

    Function *OlDestroyArgsFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_ol_destroy_" + F.getName() + Twine(taskNum)).str(),
                               {TaskArgsTy->getPointerTo()}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);
    olCallToUnpack(M, TI, TaskArgsToStructIdxMap, OlDestroyArgsFuncVar, UnpackDestroyArgsFuncVar);

    // nanos6_ol_destroy_* END

    // nanos6_ol_duplicate_* START

    Function *OlDuplicateArgsFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_ol_duplicate_" + F.getName() + Twine(taskNum)).str(),
                               {TaskArgsTy->getPointerTo(), TaskArgsTy->getPointerTo()->getPointerTo()}, {"task_args_src", "task_args_dst"},
                               {}, {});
    duplicateArgs(M, TI, TaskArgsToStructIdxMap, OlDuplicateArgsFuncVar, TaskArgsTy);

    // nanos6_ol_duplicate_* END

    if (!TI.LoopInfo.empty()) {
      TaskExtraTypeList.push_back(
        Nanos6LoopBounds::getInstance(M).getType()->getPointerTo());
      TaskExtraNameList.push_back("loop_bounds");
    } else {
      // void *device_env
      TaskExtraTypeList.push_back(Type::getInt8PtrTy(M.getContext()));
      TaskExtraNameList.push_back("device_env");
    }
    // nanos6_address_translation_entry_t *address_translation_table
    TaskExtraTypeList.push_back(
      Nanos6TaskAddrTranslationEntry::getInstance(M).getType()->getPointerTo());
    TaskExtraNameList.push_back("address_translation_table");

    // nanos6_unpacked_task_region_* START
    // CodeExtractor will create a entry block for us
    Function *UnpackTaskFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_unpacked_task_region_" + F.getName() + Twine(taskNum)).str(),
                               TaskTypeList, TaskNameList,
                               TaskExtraTypeList, TaskExtraNameList);
    // nanos6_unpacked_task_region_* END

    // nanos6_ol_task_region_* START
    Function *OlTaskFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_ol_task_region_" + F.getName() + Twine(taskNum)).str(),
                               {TaskArgsTy->getPointerTo()}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);

    olCallToUnpack(M, TI, TaskArgsToStructIdxMap, OlTaskFuncVar, UnpackTaskFuncVar, /*IsTaskFunc=*/true);

    // nanos6_ol_task_region_* END

    // nanos6_unpacked_deps_* START
    TaskExtraTypeList.clear();
    TaskExtraNameList.clear();
    // nanos6_loop_bounds_t *const loop_bounds
    TaskExtraTypeList.push_back(
      Nanos6LoopBounds::getInstance(M).getType()->getPointerTo());
    TaskExtraNameList.push_back("loop_bounds");
    // void *handler
    TaskExtraTypeList.push_back(Type::getInt8PtrTy(M.getContext()));
    TaskExtraNameList.push_back("handler");

    Function *UnpackDepsFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_unpacked_deps_" + F.getName() + Twine(taskNum)).str(),
                               TaskTypeList, TaskNameList,
                               TaskExtraTypeList, TaskExtraNameList);
    unpackDepsAndRewrite(M, TI, UnpackDepsFuncVar, TaskArgsToStructIdxMap);

    // nanos6_unpacked_deps_* END

    // nanos6_ol_deps_* START

    Function *OlDepsFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_ol_deps_" + F.getName() + Twine(taskNum)).str(),
                               {TaskArgsTy->getPointerTo()}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);
    olCallToUnpack(M, TI, TaskArgsToStructIdxMap, OlDepsFuncVar, UnpackDepsFuncVar);

    // nanos6_ol_deps_* END

    Function *OlConstraintsFuncVar = nullptr;
    if (TI.Cost) {
      // nanos6_unpacked_constraints_* START
      TaskExtraTypeList.clear();
      TaskExtraNameList.clear();
      // nanos6_task_constraints_t *constraints
      TaskExtraTypeList.push_back(Nanos6TaskConstraints::getInstance(M).getType()->getPointerTo());
      TaskExtraNameList.push_back("constraints");

      Function *UnpackConstraintsFuncVar = createUnpackOlFunction(M, F,
                                 ("nanos6_unpacked_constraints_" + F.getName() + Twine(taskNum)).str(),
                                 TaskTypeList, TaskNameList,
                                 TaskExtraTypeList, TaskExtraNameList);
      unpackCostAndRewrite(M, TI.Cost, UnpackConstraintsFuncVar, TaskArgsToStructIdxMap);
      // nanos6_unpacked_constraints_* END

      // nanos6_ol_constraints_* START

      OlConstraintsFuncVar
        = createUnpackOlFunction(M, F,
                                 ("nanos6_ol_constraints_" + F.getName() + Twine(taskNum)).str(),
                                 {TaskArgsTy->getPointerTo()}, {"task_args"},
                                 TaskExtraTypeList, TaskExtraNameList);
      olCallToUnpack(M, TI, TaskArgsToStructIdxMap, OlConstraintsFuncVar, UnpackConstraintsFuncVar);

      // nanos6_ol_constraints_* END
    }

    Function *OlPriorityFuncVar = nullptr;
    if (TI.Priority) {
      // nanos6_unpacked_priority_* START
      TaskExtraTypeList.clear();
      TaskExtraNameList.clear();
      // nanos6_priority_t *priority
      // long int *priority
      TaskExtraTypeList.push_back(Type::getInt64Ty(M.getContext())->getPointerTo());
      TaskExtraNameList.push_back("priority");

      Function *UnpackPriorityFuncVar = createUnpackOlFunction(M, F,
                                 ("nanos6_unpacked_priority_" + F.getName() + Twine(taskNum)).str(),
                                 TaskTypeList, TaskNameList,
                                 TaskExtraTypeList, TaskExtraNameList);
      unpackPriorityAndRewrite(M, TI.Priority, UnpackPriorityFuncVar, TaskArgsToStructIdxMap);
      // nanos6_unpacked_priority_* END

      // nanos6_ol_priority_* START

      OlPriorityFuncVar
        = createUnpackOlFunction(M, F,
                                 ("nanos6_ol_priority_" + F.getName() + Twine(taskNum)).str(),
                                 {TaskArgsTy->getPointerTo()}, {"task_args"},
                                 TaskExtraTypeList, TaskExtraNameList);
      olCallToUnpack(M, TI, TaskArgsToStructIdxMap, OlPriorityFuncVar, UnpackPriorityFuncVar);

      // nanos6_ol_priority_* END
    }

    // 3. Create Nanos6 task data structures info
    Constant *TaskInvInfoVar = M.getOrInsertGlobal(("task_invocation_info_" + F.getName() + Twine(taskNum)).str(),
                                      Nanos6TaskInvInfo::getInstance(M).getType(),
                                      [&M, &F, &Nanos6TaskLocStr, &taskNum] {
      GlobalVariable *GV = new GlobalVariable(M, Nanos6TaskInvInfo::getInstance(M).getType(),
                                /*isConstant=*/true,
                                GlobalVariable::InternalLinkage,
                                ConstantStruct::get(Nanos6TaskInvInfo::getInstance(M).getType(),
                                                    Nanos6TaskLocStr),
                                ("task_invocation_info_" + F.getName() + Twine(taskNum)).str());
      GV->setAlignment(Align(64));
      return GV;
    });

    Constant *TaskImplInfoVar = M.getOrInsertGlobal(("implementations_var_" + F.getName() + Twine(taskNum)).str(),
                                      ArrayType::get(Nanos6TaskImplInfo::getInstance(M).getType(), 1),
                                      [&M, &F, &OlTaskFuncVar,
                                       &OlConstraintsFuncVar, &Nanos6TaskLocStr,
                                       &TI,
                                       &taskNum] {
      bool IsConstLabelOrNull = !TI.Label || isa<Constant>(TI.Label);
      GlobalVariable *GV = new GlobalVariable(M, ArrayType::get(Nanos6TaskImplInfo::getInstance(M).getType(), 1),
                                /*isConstant=*/IsConstLabelOrNull,
                                GlobalVariable::InternalLinkage,
                                ConstantArray::get(ArrayType::get(Nanos6TaskImplInfo::getInstance(M).getType(), 1), // TODO: More than one implementations?
                                                   ConstantStruct::get(Nanos6TaskImplInfo::getInstance(M).getType(),
                                                                       ConstantInt::get(Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(0), 0),
                                                                       ConstantExpr::getPointerCast(OlTaskFuncVar, Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(1)),
                                                                       OlConstraintsFuncVar
                                                                         ? ConstantExpr::getPointerCast(OlConstraintsFuncVar,
                                                                                                        Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(2))
                                                                         : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(2))),
                                                                       TI.Label && isa<Constant>(TI.Label)
                                                                         ? ConstantExpr::getPointerCast(cast<Constant>(TI.Label),
                                                                                                        Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(3))
                                                                         : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(3))),
                                                                       Nanos6TaskLocStr,
                                                                       ConstantPointerNull::get(cast<PointerType>(Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(5))))),
                                ("implementations_var_" + F.getName() + Twine(taskNum)).str());

      GV->setAlignment(Align(64));
      return GV;
    });

    Constant *TaskRedInitsVar = M.getOrInsertGlobal(("nanos6_reduction_initializers_" + F.getName() + Twine(taskNum)).str(),
                                      ArrayType::get(FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo(), TI.ReductionsInitCombInfo.size()),
                                      [&M, &F, &TI ,&taskNum] {
      SmallVector<Constant *, 4> Inits;
      for (auto &p : TI.ReductionsInitCombInfo) {
        Inits.push_back(ConstantExpr::getPointerCast(cast<Constant>(p.second.Init),
                        FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo()));
      }

      GlobalVariable *GV = new GlobalVariable(M, ArrayType::get(FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo(), TI.ReductionsInitCombInfo.size()),
                                              /*isConstant=*/true,
                                              GlobalVariable::InternalLinkage,
                                              ConstantArray::get(ArrayType::get(FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo(), TI.ReductionsInitCombInfo.size()),
                                                                 Inits),
                                ("nanos6_reduction_initializers_" + F.getName() + Twine(taskNum)).str());
      return GV;
    });

    Constant *TaskRedCombsVar = M.getOrInsertGlobal(("nanos6_reduction_combiners_" + F.getName() + Twine(taskNum)).str(),
                                      ArrayType::get(FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo(), TI.ReductionsInitCombInfo.size()),
                                      [&M, &F, &TI, &taskNum] {
      SmallVector<Constant *, 4> Combs;
      for (auto &p : TI.ReductionsInitCombInfo) {
        Combs.push_back(ConstantExpr::getPointerCast(cast<Constant>(p.second.Comb),
                        FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo()));
      }

      GlobalVariable *GV = new GlobalVariable(M, ArrayType::get(FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo(), TI.ReductionsInitCombInfo.size()),
                                              /*isConstant=*/true,
                                              GlobalVariable::InternalLinkage,
                                              ConstantArray::get(ArrayType::get(FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo(), TI.ReductionsInitCombInfo.size()),
                                                                 Combs),
                                ("nanos6_reduction_combiners_" + F.getName() + Twine(taskNum)).str());
      return GV;
    });

    Constant *TaskInfoVar = M.getOrInsertGlobal(("task_info_var_" + F.getName() + Twine(taskNum)).str(),
                                      Nanos6TaskInfo::getInstance(M).getType(),
                                      [&M, &F, &TI, &OlDepsFuncVar,
                                       &OlPriorityFuncVar,
                                       &TaskImplInfoVar,
                                       &OlDestroyArgsFuncVar,
                                       &OlDuplicateArgsFuncVar,
                                       &TaskRedInitsVar, &TaskRedCombsVar,
                                       &taskNum] {
      GlobalVariable *GV = new GlobalVariable(M, Nanos6TaskInfo::getInstance(M).getType(),
                                /*isConstant=*/false,
                                GlobalVariable::InternalLinkage,
                                ConstantStruct::get(Nanos6TaskInfo::getInstance(M).getType(),
                                                    // TODO: Add support for devices
                                                    ConstantInt::get(Nanos6TaskInfo::getInstance(M).getType()->getElementType(0), TI.DependsInfo.NumSymbols),
                                                    ConstantExpr::getPointerCast(OlDepsFuncVar, Nanos6TaskInfo::getInstance(M).getType()->getElementType(1)),
                                                    OlPriorityFuncVar
                                                      ? ConstantExpr::getPointerCast(OlPriorityFuncVar,
                                                                                     Nanos6TaskInfo::getInstance(M).getType()->getElementType(2))
                                                      : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(2))),
                                                    ConstantInt::get(Nanos6TaskInfo::getInstance(M).getType()->getElementType(3), 1),
                                                    ConstantExpr::getPointerCast(TaskImplInfoVar, Nanos6TaskInfo::getInstance(M).getType()->getElementType(4)),
                                                    ConstantExpr::getPointerCast(OlDestroyArgsFuncVar, cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(5))),
                                                    ConstantExpr::getPointerCast(OlDuplicateArgsFuncVar, cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(6))),
                                                    ConstantExpr::getPointerCast(TaskRedInitsVar, cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(7))),
                                                    ConstantExpr::getPointerCast(TaskRedCombsVar, cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(8))),
                                                    ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(9)))),
                                ("task_info_var_" + F.getName() + Twine(taskNum)).str());

      GV->setAlignment(Align(64));
      return GV;
    });
    registerTaskInfo(M, TaskInfoVar);

    auto rewriteUsesBrAndGetOmpSsUnpackFunc
      = [&M, &TI, &EntryBB, &NewLoopInfo, &UnpackTaskFuncVar, &TaskArgsToStructIdxMap](BasicBlock *header,
                                            BasicBlock *newRootNode,
                                            BasicBlock *newHeader,
                                            Function *oldFunction,
                                            const SetVector<BasicBlock *> &Blocks) {
      UnpackTaskFuncVar->getBasicBlockList().push_back(newRootNode);

      if (!TI.LoopInfo.empty()) {
        Type *IndVarTy = TI.LoopInfo.IndVar->getType()->getPointerElementType();

        IRBuilder<> IRB(&header->front());
        Value *LoopBounds = &*(UnpackTaskFuncVar->arg_end() - 2);

        Value *Idx[2];
        Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
        Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
        Value *LBoundField = IRB.CreateGEP(LoopBounds, Idx, "lb_gep");
        LBoundField = IRB.CreateLoad(LBoundField);
        LBoundField = IRB.CreateSExtOrTrunc(LBoundField, IndVarTy, "lb");

        Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 1);
        Value *UBoundField = IRB.CreateGEP(LoopBounds, Idx, "ub_gep");
        UBoundField = IRB.CreateLoad(UBoundField);
        UBoundField = IRB.CreateSExtOrTrunc(UBoundField, IndVarTy, "ub");

        // Replace loop bounds of the indvar, loop cond. and loop incr.
        if (isReplaceableValue(NewLoopInfo.LBound)) {
          rewriteUsesInBlocksWithPred(
            NewLoopInfo.LBound, LBoundField,
            [&Blocks](Instruction *I) { return Blocks.count(I->getParent()); });
        }
        if (isReplaceableValue(NewLoopInfo.UBound)) {
          rewriteUsesInBlocksWithPred(
            NewLoopInfo.UBound, UBoundField,
            [&Blocks](Instruction *I) { return Blocks.count(I->getParent()); });
        }

        // Now we can set BodyIndVar = (LoopIndVar * Step) + OrigLBound
        IRBuilder<> LoopBodyIRB(&EntryBB->front());
        Value *NormVal = LoopBodyIRB.CreateLoad(NewLoopInfo.IndVar);
        NormVal = LoopBodyIRB.CreateMul(NormVal, IRB.CreateSExtOrTrunc(NewLoopInfo.Step, IndVarTy));
        NormVal = LoopBodyIRB.CreateAdd(NormVal, IRB.CreateSExtOrTrunc(NewLoopInfo.LBound, IndVarTy));
        LoopBodyIRB.CreateStore(NormVal, TI.LoopInfo.IndVar);
      }

      // Create an iterator to name all of the arguments we inserted.
      Function::arg_iterator AI = UnpackTaskFuncVar->arg_begin();
      // Rewrite all users of the TaskArgsToStructIdxMap in the extracted region to use the
      // arguments (or appropriate addressing into struct) instead.
      for (auto It = TaskArgsToStructIdxMap.begin();
             It != TaskArgsToStructIdxMap.end(); ++It) {
        Value *RewriteVal = &*AI++;
        Value *Val = It->first;

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
    auto emitOmpSsCaptureAndSubmitTask
      = [this, &M, &DLoc, &TaskArgsTy,
         &TI, &TaskArgsToStructIdxMap,
         &TaskInfoVar, &TaskImplInfoVar,
         &TaskInvInfoVar](Function *newFunction,
                          BasicBlock *codeReplacer,
                          const SetVector<BasicBlock *> &Blocks) {

      IRBuilder<> IRB(codeReplacer);
      // Set debug info from the task entry to all instructions
      IRB.SetCurrentDebugLocation(DLoc);

      AllocaInst *TaskArgsVar = IRB.CreateAlloca(TaskArgsTy->getPointerTo());
      Value *TaskArgsVarCast = IRB.CreateBitCast(TaskArgsVar, IRB.getInt8PtrTy()->getPointerTo());
      // typedef enum {
      //         //! Specifies that the task will be a final task
      //         nanos6_final_task = (1 << 0),
      //         //! Specifies that the task is in "if(0)" mode → !If
      //         nanos6_if_0_task = (1 << 1),
      //         //! Specifies that the task is really a taskloop
      //         nanos6_taskloop_task = (1 << 2),
      //         //! Specifies that the task is really a taskfor
      //         nanos6_taskfor_task = (1 << 3),
      //         //! Specifies that the task has the "wait" clause
      //         nanos6_waiting_task = (1 << 4),
      //         //! Specifies that the args_block is preallocated from user side
      //         nanos6_preallocated_args_block = (1 << 5),
      //         //! Specifies that the task has been verified by the user, hence it doesn't need runtime linting
      //         nanos6_verified_task = (1 << 6)
      // } nanos6_task_flag_t;
      Value *TaskFlagsVar = ConstantInt::get(IRB.getInt64Ty(), 0);
      if (TI.Final) {
        TaskFlagsVar =
          IRB.CreateOr(
            TaskFlagsVar,
            IRB.CreateZExt(TI.Final,
                           IRB.getInt64Ty()));
      }
      if (TI.If) {
        TaskFlagsVar =
          IRB.CreateOr(
            TaskFlagsVar,
            IRB.CreateShl(
              IRB.CreateZExt(
                IRB.CreateICmpEQ(TI.If, IRB.getFalse()),
                IRB.getInt64Ty()),
                1));
      }
      if (!TI.LoopInfo.empty()) {
        if (TI.TaskKind == TaskInfo::OSSD_taskloop
            || TI.TaskKind == TaskInfo::OSSD_taskloop_for) {
          TaskFlagsVar =
            IRB.CreateOr(
              TaskFlagsVar,
              IRB.CreateShl(
                  ConstantInt::get(IRB.getInt64Ty(), 1),
                  2));
        }
        if (TI.TaskKind == TaskInfo::OSSD_task_for
            || TI.TaskKind == TaskInfo::OSSD_taskloop_for) {
          TaskFlagsVar =
            IRB.CreateOr(
              TaskFlagsVar,
              IRB.CreateShl(
                  ConstantInt::get(IRB.getInt64Ty(), 1),
                  3));
        }
      }
      if (TI.Wait) {
        TaskFlagsVar =
          IRB.CreateOr(
            TaskFlagsVar,
            IRB.CreateShl(
              IRB.CreateZExt(
                TI.Wait,
                IRB.getInt64Ty()),
                4));
      }
      Value *TaskPtrVar = IRB.CreateAlloca(IRB.getInt8PtrTy());

      Value *TaskArgsStructSizeOf = ConstantInt::get(IRB.getInt64Ty(), M.getDataLayout().getTypeAllocSize(TaskArgsTy));

      // TODO: this forces an alignment of 16 for VLAs
      {
        const int ALIGN = 16;
        TaskArgsStructSizeOf =
          IRB.CreateNUWAdd(TaskArgsStructSizeOf,
                           ConstantInt::get(IRB.getInt64Ty(), ALIGN - 1));
        TaskArgsStructSizeOf =
          IRB.CreateAnd(TaskArgsStructSizeOf,
                        IRB.CreateNot(ConstantInt::get(IRB.getInt64Ty(), ALIGN - 1)));
      }

      Value *TaskArgsVLAsExtraSizeOf = computeTaskArgsVLAsExtraSizeOf(M, IRB, TI.VLADimsInfo);
      Value *TaskArgsSizeOf = IRB.CreateNUWAdd(TaskArgsStructSizeOf, TaskArgsVLAsExtraSizeOf);
      int NumDependencies =
        TI.DependsInfo.Ins.size()
        + TI.DependsInfo.Outs.size()
        + TI.DependsInfo.Inouts.size()
        + TI.DependsInfo.Concurrents.size()
        + TI.DependsInfo.Commutatives.size()
        + TI.DependsInfo.WeakIns.size()
        + TI.DependsInfo.WeakOuts.size()
        + TI.DependsInfo.WeakInouts.size()
        + TI.DependsInfo.WeakConcurrents.size()
        + TI.DependsInfo.WeakCommutatives.size()
        + TI.DependsInfo.Reductions.size()
        + TI.DependsInfo.WeakReductions.size();

      // If taskloop NumDeps = -1
      if (!TI.LoopInfo.empty() &&
          (TI.TaskKind == TaskInfo::OSSD_taskloop
           || TI.TaskKind == TaskInfo::OSSD_taskloop_for))
        NumDependencies = -1;

      // Store label if it's not a string literal (i.e label("L1"))
      if (TI.Label && !isa<Constant>(TI.Label)) {
        Value *Idx[3];
        Idx[0] = Constant::getNullValue(Type::getInt32Ty(IRB.getContext()));
        Idx[1] = Constant::getNullValue(Type::getInt32Ty(IRB.getContext()));
        Idx[2] = ConstantInt::get(Type::getInt32Ty(IRB.getContext()), 3);
        Value *LabelField = IRB.CreateGEP(TaskImplInfoVar, Idx, "ASDF");
        IRB.CreateStore(TI.Label, LabelField);
      }

      IRB.CreateCall(CreateTaskFuncCallee, {TaskInfoVar,
                                  TaskInvInfoVar,
                                  TaskArgsSizeOf,
                                  TaskArgsVarCast,
                                  TaskPtrVar,
                                  TaskFlagsVar,
                                  ConstantInt::get(IRB.getInt64Ty(),
                                                   NumDependencies)});

      // DSA capture
      Value *TaskArgsVarL = IRB.CreateLoad(TaskArgsVar);

      Value *TaskArgsVarLi8 = IRB.CreateBitCast(TaskArgsVarL, IRB.getInt8PtrTy());
      Value *TaskArgsVarLi8IdxGEP = IRB.CreateGEP(TaskArgsVarLi8, TaskArgsStructSizeOf, "args_end");

      SmallVector<VLAAlign, 2> VLAAlignsInfo;
      computeVLAsAlignOrder(M, VLAAlignsInfo, TI.VLADimsInfo);

      // First point VLAs to its according space in task args
      for (const VLAAlign& VAlign : VLAAlignsInfo) {
        Value *const V = VAlign.V;
        unsigned TyAlign = VAlign.Align;

        Type *Ty = V->getType()->getPointerElementType();

        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsToStructIdxMap[V]);
        Value *GEP = IRB.CreateGEP(
            TaskArgsVarL, Idx, "gep_" + V->getName());

        // Point VLA in task args to an aligned position of the extra space allocated
        Value *GEPi8 = IRB.CreateBitCast(GEP, IRB.getInt8PtrTy()->getPointerTo());
        IRB.CreateAlignedStore(TaskArgsVarLi8IdxGEP, GEPi8, Align(TyAlign));
        // Skip current VLA size
        unsigned SizeB = M.getDataLayout().getTypeAllocSize(Ty);
        Value *VLASize = ConstantInt::get(IRB.getInt64Ty(), SizeB);
        for (Value *const &Dim : TI.VLADimsInfo[V])
          VLASize = IRB.CreateNUWMul(VLASize, Dim);
        TaskArgsVarLi8IdxGEP = IRB.CreateGEP(TaskArgsVarLi8IdxGEP, VLASize);
      }

      for (Value *V : TI.DSAInfo.Shared) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsToStructIdxMap[V]);
        Value *GEP = IRB.CreateGEP(
            TaskArgsVarL, Idx, "gep_" + V->getName());
        IRB.CreateStore(V, GEP);
      }
      for (Value *V : TI.DSAInfo.Private) {
        // Call custom constructor generated in clang in non-pods
        // Leave pods unititialized
        auto It = TI.NonPODsInfo.Inits.find(V);
        if (It != TI.NonPODsInfo.Inits.end()) {
          Type *Ty = V->getType()->getPointerElementType();
          // Compute num elements
          Value *NSize = ConstantInt::get(IRB.getInt64Ty(), 1);
          if (isa<ArrayType>(Ty)) {
            while (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
              // Constant array
              Value *NumElems = ConstantInt::get(IRB.getInt64Ty(),
                                                 ArrTy->getNumElements());
              NSize = IRB.CreateNUWMul(NSize, NumElems);
              Ty = ArrTy->getElementType();
            }
          } else if (TI.VLADimsInfo.count(V)) {
            for (Value *const &Dim : TI.VLADimsInfo[V])
              NSize = IRB.CreateNUWMul(NSize, Dim);
          }

          Value *Idx[2];
          Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
          Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsToStructIdxMap[V]);
          Value *GEP = IRB.CreateGEP(
              TaskArgsVarL, Idx, "gep_" + V->getName());

          // VLAs
          if (TI.VLADimsInfo.count(V))
            GEP = IRB.CreateLoad(GEP);

          // Regular arrays have types like [10 x %struct.S]*
          // Cast to %struct.S*
          GEP = IRB.CreateBitCast(GEP, Ty->getPointerTo());

          IRB.CreateCall(FunctionCallee(cast<Function>(It->second)), ArrayRef<Value*>{GEP, NSize});
        }
      }
      for (Value *V : TI.DSAInfo.Firstprivate) {
        Type *Ty = V->getType()->getPointerElementType();
        unsigned TyAlign = M.getDataLayout().getPrefTypeAlignment(Ty);

        // Compute num elements
        Value *NSize = ConstantInt::get(IRB.getInt64Ty(), 1);
        if (isa<ArrayType>(Ty)) {
          while (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
            // Constant array
            Value *NumElems = ConstantInt::get(IRB.getInt64Ty(),
                                               ArrTy->getNumElements());
            NSize = IRB.CreateNUWMul(NSize, NumElems);
            Ty = ArrTy->getElementType();
          }
        } else if (TI.VLADimsInfo.count(V)) {
          for (Value *const &Dim : TI.VLADimsInfo[V])
            NSize = IRB.CreateNUWMul(NSize, Dim);
        }

        // call custom copy constructor generated in clang in non-pods
        // do a memcpy if pod
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsToStructIdxMap[V]);
        Value *GEP = IRB.CreateGEP(
            TaskArgsVarL, Idx, "gep_" + V->getName());

        // VLAs
        if (TI.VLADimsInfo.count(V))
          GEP = IRB.CreateLoad(GEP);

        auto It = TI.NonPODsInfo.Copies.find(V);
        if (It != TI.NonPODsInfo.Copies.end()) {
          // Non-POD

          // Regular arrays have types like [10 x %struct.S]*
          // Cast to %struct.S*
          GEP = IRB.CreateBitCast(GEP, Ty->getPointerTo());
          V = IRB.CreateBitCast(V, Ty->getPointerTo());

          llvm::Function *Func = cast<Function>(It->second);
          IRB.CreateCall(Func, ArrayRef<Value*>{/*Src=*/V, /*Dst=*/GEP, NSize});
        } else {
          unsigned SizeB = M.getDataLayout().getTypeAllocSize(Ty);
          Value *NSizeB = IRB.CreateNUWMul(NSize, ConstantInt::get(IRB.getInt64Ty(), SizeB));
          IRB.CreateMemCpy(GEP, Align(TyAlign), V, Align(TyAlign), NSizeB);
        }
      }
      for (Value *V : TI.CapturedInfo) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsToStructIdxMap[V]);
        Value *GEP = IRB.CreateGEP(
            TaskArgsVarL, Idx, "capt_gep_" + V->getName());
        IRB.CreateStore(V, GEP);
      }

      Value *TaskPtrVarL = IRB.CreateLoad(TaskPtrVar);

      if (!TI.LoopInfo.empty()) {
        // <      0, (ub - 1 - lb) / step + 1
        // <=     0, (ub - lb)     / step + 1
        // >      0, (ub + 1 - lb) / step + 1
        // >=     0, (ub - lb)     / step + 1
        Type *IndVarTy = TI.LoopInfo.IndVar->getType()->getPointerElementType();
        Value *RegisterLowerB = ConstantInt::get(IndVarTy, 0);
        Value *RegisterUpperB = IRB.CreateSExtOrTrunc(TI.LoopInfo.UBound, IndVarTy);
        RegisterUpperB = IRB.CreateSub(RegisterUpperB, IRB.CreateSExtOrTrunc(TI.LoopInfo.LBound, IndVarTy));
        Value *RegisterGrainsize = ConstantInt::get(IndVarTy, 0);
        if (TI.LoopInfo.Grainsize)
          RegisterGrainsize = TI.LoopInfo.Grainsize;
        Value *RegisterChunksize = ConstantInt::get(IndVarTy, 0);
        if (TI.LoopInfo.Chunksize)
          RegisterChunksize = TI.LoopInfo.Chunksize;

        switch (TI.LoopInfo.LoopType) {
        case TaskLoopInfo::SLT:
        case TaskLoopInfo::ULT:
          RegisterUpperB = IRB.CreateSub(RegisterUpperB, ConstantInt::get(IndVarTy, 1));
          break;
        case TaskLoopInfo::SGT:
        case TaskLoopInfo::UGT:
          RegisterUpperB = IRB.CreateAdd(RegisterUpperB, ConstantInt::get(IndVarTy, 1));
          break;
        case TaskLoopInfo::SLE:
        case TaskLoopInfo::ULE:
        case TaskLoopInfo::SGE:
        case TaskLoopInfo::UGE:
          break;
        default:
          llvm_unreachable("unexpected loop type");
        }
        // TODO: should we handle sdiv/udiv depending on loop type?
        RegisterUpperB = IRB.CreateSDiv(RegisterUpperB, IRB.CreateSExtOrTrunc(TI.LoopInfo.Step, IndVarTy));
        RegisterUpperB = IRB.CreateAdd(RegisterUpperB, ConstantInt::get(IndVarTy, 1));
        IRB.CreateCall(
          RegisterLoopFuncCallee,
          {
            TaskPtrVarL,
            IRB.CreateSExt(RegisterLowerB, Nanos6LoopBounds::getInstance(M).getType()->getElementType(0)),
            IRB.CreateSExt(RegisterUpperB, Nanos6LoopBounds::getInstance(M).getType()->getElementType(1)),
            IRB.CreateSExt(RegisterGrainsize, Nanos6LoopBounds::getInstance(M).getType()->getElementType(2)),
            IRB.CreateSExt(RegisterChunksize, Nanos6LoopBounds::getInstance(M).getType()->getElementType(3))
          }
        );
      }

      CallInst *TaskSubmitFuncCall = IRB.CreateCall(TaskSubmitFuncCallee, TaskPtrVarL);
      // Add a branch to the next basic block after the task region
      // and replace the terminator that exits the task region
      // Since this is a single entry single exit region this should
      // be done once.
      BasicBlock *NewRetBB = nullptr;
      for (BasicBlock *Block : Blocks) {
        Instruction *TI = Block->getTerminator();
        for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
          if (!Blocks.count(TI->getSuccessor(i))) {
            assert(!NewRetBB && "More than one exit in task code");

            BasicBlock *OldTarget = TI->getSuccessor(i);
            // Create branch to next BB after the task region
            IRB.CreateBr(OldTarget);

            NewRetBB = BasicBlock::Create(M.getContext(), ".exitStub", newFunction);
            IRBuilder<> (NewRetBB).CreateRetVoid();

            // rewrite the original branch instruction with this new target
            TI->setSuccessor(i, NewRetBB);
          }
      }

      return TaskSubmitFuncCall;
    };

    // 4. Extract region the way we want
    CodeExtractorAnalysisCache CEAC(F);
    CodeExtractor CE(TaskBBs.getArrayRef(), rewriteUsesBrAndGetOmpSsUnpackFunc, emitOmpSsCaptureAndSubmitTask);
    CE.extractCodeRegion(CEAC);
  }

  void buildNanos6Types(Module &M) {
    // void nanos6_create_task(
    //         nanos6_task_info_t *task_info,
    //         nanos6_task_invocation_info_t *task_invocation_info,
    //         size_t args_block_size,
    //         /* OUT */ void **args_block_pointer,
    //         /* OUT */ void **task_pointer,
    //         size_t flags,
    //         size_t num_deps
    // );
    CreateTaskFuncCallee = M.getOrInsertFunction("nanos6_create_task",
        Type::getVoidTy(M.getContext()),
        Nanos6TaskInfo::getInstance(M).getType()->getPointerTo(),
        Nanos6TaskInvInfo::getInstance(M).getType()->getPointerTo(),
        Type::getInt64Ty(M.getContext()),
        Type::getInt8PtrTy(M.getContext())->getPointerTo(),
        Type::getInt8PtrTy(M.getContext())->getPointerTo(),
        Type::getInt64Ty(M.getContext()),
        Type::getInt64Ty(M.getContext())
    );

    // void nanos6_submit_task(void *task);
    TaskSubmitFuncCallee = M.getOrInsertFunction("nanos6_submit_task",
        Type::getVoidTy(M.getContext()),
        Type::getInt8PtrTy(M.getContext())
    );

    // int nanos6_in_final(void);
    TaskInFinalFuncCallee= M.getOrInsertFunction("nanos6_in_final",
        Type::getInt32Ty(M.getContext())
    );

    // void nanos6_register_loop_bounds(
    //     void *taskHandle, size_t lower_bound, size_t upper_bound,
    //     size_t grainsize, size_t chunksize)
    RegisterLoopFuncCallee = M.getOrInsertFunction("nanos6_register_loop_bounds",
        Type::getVoidTy(M.getContext()),
        Type::getInt8PtrTy(M.getContext()),
        Type::getInt64Ty(M.getContext()),
        Type::getInt64Ty(M.getContext()),
        Type::getInt64Ty(M.getContext()),
        Type::getInt64Ty(M.getContext())
    );

    // void nanos6_register_task_info(nanos6_task_info_t *task_info);
    TaskInfoRegisterFuncCallee = M.getOrInsertFunction("nanos6_register_task_info",
        Type::getVoidTy(M.getContext()),
        Nanos6TaskInfo::getInstance(M).getType()->getPointerTo()
    );

    // void nanos6_constructor_register_task_info(void);
    // NOTE: This does not belong to nanos6 API
    TaskInfoRegisterCtorFuncCallee =
      M.getOrInsertFunction("nanos6_constructor_register_task_info",
        Type::getVoidTy(M.getContext())
      );
    cast<Function>(TaskInfoRegisterCtorFuncCallee.getCallee())->setLinkage(GlobalValue::InternalLinkage);
    BasicBlock *EntryBB = BasicBlock::Create(M.getContext(), "entry",
      cast<Function>(TaskInfoRegisterCtorFuncCallee.getCallee()));
    EntryBB->getInstList().push_back(ReturnInst::Create(M.getContext()));

    appendToGlobalCtors(M, cast<Function>(TaskInfoRegisterCtorFuncCallee.getCallee()), 65535);
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    // Keep all the functions before start outlining
    // to avoid analize them.
    SmallVector<Function *, 4> Functs;
    for (auto &F : M) {
      // Nothing to do for declarations.
      if (F.isDeclaration() || F.empty())
        continue;

      Functs.push_back(&F);
    }

    if (!Initialized) {
      Initialized = true;
      buildNanos6Types(M);
    }

    for (auto *F : Functs) {
      FunctionInfo &FI = getAnalysis<OmpSsRegionAnalysisPass>(*F).getFuncInfo();
      TaskFunctionInfo &TFI = FI.TaskFuncInfo;
      TaskwaitFunctionInfo &TwFI = FI.TaskwaitFuncInfo;

      DenseMap<TaskInfo *, SmallVector<FinalBodyInfo, 4>> TaskFinalInfo;

      // First sweep to clone BBs
      for (TaskInfo *pTI : TFI.PostOrder) {
        TaskInfo &TI = *pTI;
        // 1. Split BB
        BasicBlock *EntryBB = TI.Entry->getParent();
        EntryBB = EntryBB->splitBasicBlock(TI.Entry);

        BasicBlock *ExitBB = TI.Exit->getParent();
        ExitBB = ExitBB->splitBasicBlock(TI.Exit->getNextNode());

        SmallVector<BasicBlock*, 8> Worklist;
        SmallPtrSet<BasicBlock*, 8> Visited;
        DenseMap<BasicBlock *, BasicBlock *> CopyBBs;
        ValueToValueMapTy VMap;

        // 2. Clone BBs between entry and exit (is there any function/util to do this?)
        Worklist.push_back(EntryBB);
        Visited.insert(EntryBB);

        CopyBBs[EntryBB] = CloneBasicBlock(EntryBB, VMap, ".clone", F);
        VMap[EntryBB] = CopyBBs[EntryBB];
        while (!Worklist.empty()) {
          auto WIt = Worklist.begin();
          BasicBlock *BB = *WIt;
          Worklist.erase(WIt);

          for (auto It = succ_begin(BB); It != succ_end(BB); ++It) {
            if (!Visited.count(*It) && *It != ExitBB) {
              Worklist.push_back(*It);
              Visited.insert(*It);

              CopyBBs[*It] = CloneBasicBlock(*It, VMap, ".clone", F);
              VMap[*It] = CopyBBs[*It];
            }
          }
        }

        Instruction *OrigEntry = TI.Entry;
        Instruction *OrigExit = TI.Exit;
        Instruction *CloneEntry = cast<Instruction>(VMap[OrigEntry]);
        Instruction *CloneExit = cast<Instruction>(VMap[OrigExit]);
        TaskFinalInfo[&TI].push_back({CloneEntry, CloneExit, &TI});
        for (TaskInfo *InnerTI : TI.InnerTaskInfos) {
          OrigEntry = InnerTI->Entry;
          OrigExit = InnerTI->Exit;
          CloneEntry = cast<Instruction>(VMap[OrigEntry]);
          CloneExit = cast<Instruction>(VMap[OrigExit]);
          TaskFinalInfo[&TI].push_back({CloneEntry, CloneExit, &TI});
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
      }

      for (TaskwaitInfo& TwI : TwFI.PostOrder) {
        lowerTaskwait(TwI, M);
      }

      size_t taskNum = 0;
      for (TaskInfo *pTI : TFI.PostOrder) {
        TaskInfo &TI = *pTI;
        lowerTask(TI, *F, taskNum++, M, TaskFinalInfo);
      }

    }
    return true;
  }

  StringRef getPassName() const override { return "Nanos6 Lowering"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<OmpSsRegionAnalysisPass>();
  }

};

}

char OmpSs::ID = 0;

ModulePass *llvm::createOmpSsPass() {
  return new OmpSs();
}

void LLVMOmpSsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createOmpSsPass());
}

INITIALIZE_PASS_BEGIN(OmpSs, "ompss-2",
                "Transforms OmpSs-2 llvm.directive.region intrinsics", false, false)
INITIALIZE_PASS_DEPENDENCY(OmpSsRegionAnalysisPass)
INITIALIZE_PASS_END(OmpSs, "ompss-2",
                "Transforms OmpSs-2 llvm.directive.region intrinsics", false, false)
