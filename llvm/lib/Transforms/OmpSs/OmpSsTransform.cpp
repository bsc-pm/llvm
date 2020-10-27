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

    static StringRef getDependTypeStrFromType(
        DependInfo::DependType DType) {
      switch (DType) {
      case DependInfo::DT_in:
        return "read";
      case DependInfo::DT_out:
        return "write";
      case DependInfo::DT_inout:
        return "readwrite";
      case DependInfo::DT_concurrent:
        return "concurrent";
      case DependInfo::DT_commutative:
        return "commutative";
      case DependInfo::DT_reduction:
        return "reduction";
      case DependInfo::DT_weakin:
        return "weak_read";
      case DependInfo::DT_weakout:
        return "weak_write";
      case DependInfo::DT_weakinout:
        return "weak_readwrite";
      case DependInfo::DT_weakconcurrent:
        return "weak_concurrent";
      case DependInfo::DT_weakcommutative:
        return "weak_commutative";
      case DependInfo::DT_weakreduction:
        return "weak_reduction";
      default:
        break;
      }
      llvm_unreachable("unknown depend type");
    }

    FunctionType *BuildDepFuncType(
        Module &M, StringRef FullName, size_t Ndims, bool IsReduction) {
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

    FunctionType *BuildReleaseDepFuncType(
        Module &M, StringRef FullName, size_t Ndims) {
      // void nanos6_release_x_Y(
      //   void *base_address,
      //   long dim1size, long dim1start, long dim1end,
      //   ...);

      SmallVector<Type *, 8> Params;
      Params.push_back(Type::getInt8PtrTy(M.getContext()));
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
    FunctionCallee getMultidepFuncCallee(
        Module &M, DependInfo::DependType DType, size_t Ndims, bool IsReduction=false) {
      std::string FullName =
        ("nanos6_register_region_" + getDependTypeStrFromType(DType) + "_depinfo" + Twine(Ndims)).str();

      auto It = DepNameToFuncCalleeMap.find(FullName);
      if (It != DepNameToFuncCalleeMap.end())
        return It->second;

      assert(Ndims <= MAX_DEP_DIMS);

      FunctionType *DepF = BuildDepFuncType(M, FullName, Ndims, IsReduction);
      FunctionCallee DepCallee = M.getOrInsertFunction(FullName, DepF);
      DepNameToFuncCalleeMap[FullName] = DepCallee;
      return DepCallee;
    }

    FunctionCallee getReleaseMultidepFuncCallee(
        Module &M, DependInfo::DependType DType, size_t Ndims) {
      std::string FullName =
        ("nanos6_release_" + getDependTypeStrFromType(DType) + "_" + Twine(Ndims)).str();

      auto It = DepNameToFuncCalleeMap.find(FullName);
      if (It != DepNameToFuncCalleeMap.end())
        return It->second;

      assert(Ndims <= MAX_DEP_DIMS);

      FunctionType *DepF = BuildReleaseDepFuncType(M, FullName, Ndims);
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
    DirectiveInfo *DirInfo;
  };
  // For each directive have all inner directives in final context.
  using FinalBodyInfoMap = DenseMap<DirectiveInfo *, SmallVector<FinalBodyInfo, 4>>;

  // Signed extension of V to type Ty
  static Value *createZSExtOrTrunc(IRBuilder<> &IRB, Value *V, Type *Ty, bool Signed) {
    if (Signed)
      return IRB.CreateSExtOrTrunc(V, Ty);
    return IRB.CreateZExtOrTrunc(V, Ty);
  }

  // Compares LHS and RHS and extends the one with lower type size. The extension
  // is based on LHSSigned/RHSSigned
  // returns the new instruction built and the signedness
  static std::pair<Value *, bool> buildInstructionSignDependent(
      IRBuilder<> &IRB, Module &M,
      Value *LHS, Value *RHS, bool LHSSigned, bool RHSSigned,
      const llvm::function_ref<Value *(IRBuilder<> &, Value *, Value *, bool)> InstrGen) {
    Type *LHSTy = LHS->getType();
    Type *RHSTy = RHS->getType();
    TypeSize LHSTySize = M.getDataLayout().getTypeSizeInBits(LHSTy);
    TypeSize RHSTySize = M.getDataLayout().getTypeSizeInBits(RHSTy);
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
                        Instruction *Exit, const DirectiveLoopInfo &LoopInfo,
                        BasicBlock *&LoopEntryBB, BasicBlock *&BodyBB) {

    IRBuilder<> IRB(Entry);
    Type *IndVarTy = LoopInfo.IndVar->getType()->getPointerElementType();

    IRB.CreateStore(createZSExtOrTrunc(IRB, LoopInfo.LBound, IndVarTy, LoopInfo.LBoundSigned), LoopInfo.IndVar);

    BasicBlock *CondBB = IRB.saveIP().getBlock()->splitBasicBlock(IRB.saveIP().getPoint());
    CondBB->setName("for.cond");

    // The new entry is the start of the loop
    LoopEntryBB = CondBB->getUniquePredecessor();

    IRB.SetInsertPoint(Entry);

    Value *IndVarVal = IRB.CreateLoad(LoopInfo.IndVar);
    Value *LoopCmp = nullptr;
    switch (LoopInfo.LoopType) {
    case DirectiveLoopInfo::LT:
      LoopCmp = buildInstructionSignDependent(
        IRB, M, IndVarVal, LoopInfo.UBound, LoopInfo.IndVarSigned, LoopInfo.UBoundSigned,
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          if (NewOpSigned)
            return IRB.CreateICmpSLT(LHS, RHS);
          return IRB.CreateICmpULT(LHS, RHS);
        }).first;
      break;
    case DirectiveLoopInfo::LE:
      LoopCmp = buildInstructionSignDependent(
        IRB, M, IndVarVal, LoopInfo.UBound, LoopInfo.IndVarSigned, LoopInfo.UBoundSigned,
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          if (NewOpSigned)
            return IRB.CreateICmpSLE(LHS, RHS);
          return IRB.CreateICmpULE(LHS, RHS);
        }).first;
      break;
    case DirectiveLoopInfo::GT:
      LoopCmp = buildInstructionSignDependent(
        IRB, M, IndVarVal, LoopInfo.UBound, LoopInfo.IndVarSigned, LoopInfo.UBoundSigned,
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          if (NewOpSigned)
            return IRB.CreateICmpSGT(LHS, RHS);
          return IRB.CreateICmpUGT(LHS, RHS);
        }).first;
      break;
    case DirectiveLoopInfo::GE:
      LoopCmp = buildInstructionSignDependent(
        IRB, M, IndVarVal, LoopInfo.UBound, LoopInfo.IndVarSigned, LoopInfo.UBoundSigned,
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          if (NewOpSigned)
            return IRB.CreateICmpSGE(LHS, RHS);
          return IRB.CreateICmpUGE(LHS, RHS);
        }).first;
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
    auto p = buildInstructionSignDependent(
      IRB, M, IndVarVal, LoopInfo.Step, LoopInfo.IndVarSigned, LoopInfo.StepSigned,
      [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
        return IRB.CreateAdd(LHS, RHS);
      });
    IRB.CreateStore(createZSExtOrTrunc(IRB, p.first, IndVarTy, p.second), LoopInfo.IndVar);
    IRB.CreateBr(CondBB);

    // Replace task end br. by a br. to for.incr
    OldTerminator = Exit->getParent()->getTerminator();
    assert(OldTerminator->getNumSuccessors() == 1);
    OldTerminator->setSuccessor(0, IncrBB);
  }

  BasicBlock *buildLoopForTask(
      Module &M, Function &F, Instruction *Entry,
      Instruction *Exit, const DirectiveLoopInfo &LoopInfo) {
    BasicBlock *EntryBB = nullptr;
    BasicBlock *LoopEntryBB = nullptr;
    buildLoopForTaskImpl(M, F, Entry, Exit, LoopInfo, LoopEntryBB, EntryBB);
    return LoopEntryBB;
  }

  void buildLoopForMultiDep(
      Module &M, Function &F, Instruction *Entry, Instruction *Exit,
      Value *IndVar,
      const llvm::function_ref<Value *(IRBuilder<> &)> LBoundGen,
      const llvm::function_ref<Value *(IRBuilder<> &)> RemapGen,
      const llvm::function_ref<Value *(IRBuilder<> &)> UBoundGen,
      const llvm::function_ref<Value *(IRBuilder<> &)> IncrGen) {

    IRBuilder<> IRB(Entry);
    Type *IndVarTy = IndVar->getType()->getPointerElementType();

    Value *IndVarRemap = IRB.CreateAlloca(IndVarTy, nullptr, IndVar->getName() + ".remap");

    Value *LBound = LBoundGen(IRB);
    IRB.CreateStore(LBound, IndVar);

    BasicBlock *CondBB = IRB.saveIP().getBlock()->splitBasicBlock(IRB.saveIP().getPoint());
    CondBB->setName("for.cond");

    IRB.SetInsertPoint(Entry);

    Value *UBound = UBoundGen(IRB);
    Value *IndVarVal = IRB.CreateLoad(IndVar);
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
    Value *Remap = RemapGen(IRB);
    IRB.CreateStore(Remap, IndVarRemap);

    // Replace default br. by a conditional br. to task.body or task end
    Instruction *OldTerminator = CondBB->getTerminator();
    IRB.SetInsertPoint(OldTerminator);
    IRB.CreateCondBr(LoopCmp, BodyBB, Exit->getParent()->getUniqueSuccessor());
    OldTerminator->eraseFromParent();

    BasicBlock *IncrBB = BasicBlock::Create(M.getContext(), "for.incr", &F);

    // Add a br. to for.cond
    IRB.SetInsertPoint(IncrBB);

    Value *Incr = IncrGen(IRB);
    IndVarVal = IRB.CreateLoad(IndVar);
    Value *IndVarValPlusIncr = IRB.CreateAdd(IndVarVal, Incr);
    IRB.CreateStore(IndVarValPlusIncr, IndVar);
    IRB.CreateBr(CondBB);

    // Replace task end br. by a br. to for.incr
    OldTerminator = Exit->getParent()->getTerminator();
    assert(OldTerminator->getNumSuccessors() == 1);
    OldTerminator->setSuccessor(0, IncrBB);
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
      Module &M, const DirectiveInfo &DirInfo, Function *F,
      const MapVector<Value *, size_t> &StructToIdxMap) {

    BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock &Entry = F->getEntryBlock();
    IRBuilder<> IRB(&Entry);

    const DirectiveVLADimsInfo &VLADimsInfo = DirInfo.DirEnv.VLADimsInfo;
    const DirectiveNonPODsInfo &NonPODsInfo = DirInfo.DirEnv.NonPODsInfo;

    for (const auto &DeinitMap : NonPODsInfo.Deinits) {
      auto *V = DeinitMap.first;
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
      } else if (VLADimsInfo.count(V)) {
        for (Value *Dim : VLADimsInfo.lookup(V))
          NSize = IRB.CreateNUWMul(NSize, F->getArg(StructToIdxMap.lookup(Dim)));
      }

      // Regular arrays have types like [10 x %struct.S]*
      // Cast to %struct.S*
      Value *FArg = IRB.CreateBitCast(F->getArg(StructToIdxMap.lookup(V)), Ty->getPointerTo());

      llvm::Function *Func = cast<Function>(DeinitMap.second);
      IRB.CreateCall(Func, ArrayRef<Value*>{FArg, NSize});
    }
    IRB.CreateRetVoid();
  }

  void unpackDepCallToRTImpl(
      Module &M, const DependInfo *DepInfo,
      const DirectiveReductionsInitCombInfo &DRI,
      Function *F, Value *IndVar,
      Value *NewIndVarLBound, Value *NewIndVarUBound, bool IsTaskLoop,
      CallInst *& CallComputeDepStart, CallInst *& CallComputeDepEnd,
      CallInst *& RegisterDepCall) {

    BasicBlock &Entry = F->getEntryBlock();
    Instruction &RetI = Entry.back();
    IRBuilder<> BBBuilder(&RetI);

    Function *ComputeDepFun = cast<Function>(DepInfo->ComputeDepFun);
    CallComputeDepStart = BBBuilder.CreateCall(ComputeDepFun, DepInfo->Args);
    CallComputeDepEnd = BBBuilder.CreateCall(ComputeDepFun, DepInfo->Args);
    if (IsTaskLoop) {
      assert(NewIndVarLBound && NewIndVarUBound &&
        "Expected new lb/up in taskloop dependency");
      CallComputeDepStart->replaceUsesOfWith(IndVar, NewIndVarLBound);
      CallComputeDepEnd->replaceUsesOfWith(IndVar, NewIndVarUBound);
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
      TaskDepAPICall.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()), DRI.lookup(Base).ReductionIndex));
    }
    Value *Handler = &*(F->arg_end() - 1);
    TaskDepAPICall.push_back(Handler);
    TaskDepAPICall.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()), DepInfo->SymbolIndex));
    TaskDepAPICall.push_back(ConstantPointerNull::get(Type::getInt8PtrTy(M.getContext()))); // TODO: stringify
    TaskDepAPICall.push_back(BBBuilder.CreateBitCast(Base, Type::getInt8PtrTy(M.getContext())));
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
      Module &M, const DependInfo *DepInfo,
      const DirectiveReductionsInitCombInfo &DRI,
      Function *F, Value *IndVar, Value *NewIndVarLBound, Value *NewIndVarUBound,
      bool IsTaskLoop) {

    CallInst *CallComputeDepStart;
    CallInst *CallComputeDepEnd;
    CallInst *RegisterDepCall;
    unpackDepCallToRTImpl(
      M, DepInfo, DRI, F, IndVar, NewIndVarLBound, NewIndVarUBound, IsTaskLoop,
      CallComputeDepStart, CallComputeDepEnd, RegisterDepCall);
  }

  void unpackMultiRangeCall(
      Module &M, const MultiDependInfo *MultiDepInfo,
      const DirectiveReductionsInitCombInfo &DRI,
      Function *F, Value *IndVar, Value *NewIndVarLBound, Value *NewIndVarUBound,
      bool IsTaskLoop) {

    CallInst *CallComputeDepStart;
    CallInst *CallComputeDepEnd;
    CallInst *RegisterDepCall;
    unpackDepCallToRTImpl(
      M, MultiDepInfo, DRI, F, IndVar, NewIndVarLBound, NewIndVarUBound, IsTaskLoop,
      CallComputeDepStart, CallComputeDepEnd, RegisterDepCall);

    // Build a BasicBlock conatining the compute_dep call and the dep. registration
    CallComputeDepStart->getParent()->splitBasicBlock(CallComputeDepStart);
    Instruction *AfterRegisterDepCall = RegisterDepCall->getNextNode();
    AfterRegisterDepCall->getParent()->splitBasicBlock(AfterRegisterDepCall);

    Function *ComputeMultiDepFun = MultiDepInfo->ComputeMultiDepFun;
    auto Args = MultiDepInfo->Args;

    if (IsTaskLoop) {
      std::replace(Args.begin(), Args.end(), IndVar, NewIndVarLBound);
    }

    for (size_t i = 0; i < MultiDepInfo->Iters.size(); i++) {
      Value *IndVar = MultiDepInfo->Iters[i];
      auto LBoundGen = [ComputeMultiDepFun, &Args, i](IRBuilder<> &IRB) {
        Value *ComputeMultiDepCall = IRB.CreateCall(ComputeMultiDepFun, Args);
         return IRB.CreateExtractValue(ComputeMultiDepCall, i*(3 + 1) + 0);
      };
      auto RemapGen = [ComputeMultiDepFun, &Args, i](IRBuilder<> &IRB) {
        Value *ComputeMultiDepCall = IRB.CreateCall(ComputeMultiDepFun, Args);
         return IRB.CreateExtractValue(ComputeMultiDepCall, i*(3 + 1) + 1);
      };
      auto UBoundGen = [ComputeMultiDepFun, &Args, i](IRBuilder<> &IRB) {
        Value *ComputeMultiDepCall = IRB.CreateCall(ComputeMultiDepFun, Args);
         return IRB.CreateExtractValue(ComputeMultiDepCall, i*(3 + 1) + 2);
      };
      auto IncrGen = [ComputeMultiDepFun, &Args, i](IRBuilder<> &IRB) {
        Value *ComputeMultiDepCall = IRB.CreateCall(ComputeMultiDepFun, Args);
         return IRB.CreateExtractValue(ComputeMultiDepCall, i*(3 + 1) + 3);
      };
      buildLoopForMultiDep(
          M, *F, CallComputeDepStart, RegisterDepCall, IndVar, LBoundGen, RemapGen, UBoundGen, IncrGen);
    }
  }


  void unpackDepsCallToRT(
      Module &M, const DirectiveInfo &DirInfo, Function *F,
      bool IsTaskLoop, Value *NewIndVarLBound, Value *NewIndVarUBound) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveLoopInfo &LoopInfo = DirEnv.LoopInfo;
    const DirectiveReductionsInitCombInfo &DRI = DirEnv.ReductionsInitCombInfo;
    const DirectiveDependsInfo &DependsInfo = DirEnv.DependsInfo;

    for (auto &DepInfo : DependsInfo.List) {
      if (auto *MultiDepInfo = dyn_cast<MultiDependInfo>(DepInfo.get()))
        unpackMultiRangeCall(
          M, MultiDepInfo, DRI, F,
          LoopInfo.IndVar, NewIndVarLBound, NewIndVarUBound, IsTaskLoop);
      else
        unpackDepCallToRT(
          M, DepInfo.get(), DRI, F,
          LoopInfo.IndVar, NewIndVarLBound, NewIndVarUBound, IsTaskLoop);
    }
  }

  void unpackDepsAndRewrite(Module &M, const DirectiveInfo &DirInfo,
                            Function *F,
                            const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock &Entry = F->getEntryBlock();

    // add the terminator so IRBuilder inserts just before it
    F->getEntryBlock().getInstList().push_back(ReturnInst::Create(M.getContext()));

    Value *NewIndVarLBound = nullptr;
    Value *NewIndVarUBound = nullptr;

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveLoopInfo &LoopInfo = DirEnv.LoopInfo;

    bool IsTaskLoop = DirEnv.isOmpSsTaskLoopDirective();

    if (IsTaskLoop) {
      Type *IndVarTy = LoopInfo.IndVar->getType()->getPointerElementType();

      IRBuilder<> IRB(&Entry.front());
      Value *LoopBounds = &*(F->arg_end() - 2);

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
      Value *LBoundField = IRB.CreateGEP(LoopBounds, Idx, "lb_gep");
      LBoundField = IRB.CreateLoad(LBoundField);
      LBoundField = IRB.CreateZExtOrTrunc(LBoundField, IndVarTy, "lb");

      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 1);
      Value *UBoundField = IRB.CreateGEP(LoopBounds, Idx, "ub_gep");
      UBoundField = IRB.CreateLoad(UBoundField);
      UBoundField = IRB.CreateZExtOrTrunc(UBoundField, IndVarTy);
      UBoundField = IRB.CreateSub(UBoundField, ConstantInt::get(IndVarTy, 1), "ub");

      NewIndVarLBound = IRB.CreateAlloca(IndVarTy, nullptr, LoopInfo.IndVar->getName() + ".lb");
      NewIndVarUBound = IRB.CreateAlloca(IndVarTy, nullptr, LoopInfo.IndVar->getName() + ".ub");

      auto p = buildInstructionSignDependent(
        IRB, M, LoopInfo.Step, LBoundField, LoopInfo.StepSigned, LoopInfo.IndVarSigned,
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          return IRB.CreateMul(LHS, RHS);
        });
      p = buildInstructionSignDependent(
        IRB, M, p.first, LoopInfo.LBound, p.second, LoopInfo.LBoundSigned,
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          return IRB.CreateAdd(LHS, RHS);
        });
      IRB.CreateStore(createZSExtOrTrunc(IRB, p.first, IndVarTy, p.second), NewIndVarLBound);

      p = buildInstructionSignDependent(
        IRB, M, LoopInfo.Step, UBoundField, LoopInfo.StepSigned, LoopInfo.IndVarSigned,
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          return IRB.CreateMul(LHS, RHS);
        });
      p = buildInstructionSignDependent(
        IRB, M, p.first, LoopInfo.LBound, p.second, LoopInfo.LBoundSigned,
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          return IRB.CreateAdd(LHS, RHS);
        });
      IRB.CreateStore(createZSExtOrTrunc(IRB, p.first, IndVarTy, p.second), NewIndVarUBound);
    }

    // Insert RT call before replacing uses
    unpackDepsCallToRT(M, DirInfo, F, IsTaskLoop, NewIndVarLBound, NewIndVarUBound);

    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        Function::arg_iterator AI = F->arg_begin();
        for (auto It = StructToIdxMap.begin();
               It != StructToIdxMap.end(); ++It, ++AI) {
          if (isReplaceableValue(It->first))
            I.replaceUsesOfWith(It->first, &*AI);
        }
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

  void unpackReleaseDepCallToRT(
      Module &M, const DependInfo *DepInfo, Instruction *InsertPt) {

    IRBuilder<> BBBuilder(InsertPt);

    Function *ComputeDepFun = cast<Function>(DepInfo->ComputeDepFun);
    Value *CallComputeDep = BBBuilder.CreateCall(ComputeDepFun, DepInfo->Args);
    StructType *ComputeDepTy = cast<StructType>(ComputeDepFun->getReturnType());

    assert(ComputeDepTy->getNumElements() > 1 && "Expected dependency base with dim_{size, start, end}");
    size_t NumDims = (ComputeDepTy->getNumElements() - 1)/3;

    llvm::Value *Base = BBBuilder.CreateExtractValue(CallComputeDep, 0);

    SmallVector<Value *, 4> TaskDepAPICall;
    TaskDepAPICall.push_back(BBBuilder.CreateBitCast(Base, Type::getInt8PtrTy(M.getContext())));
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

  void unpackReleaseDepsCallToRT(
      Module &M, const DirectiveInfo &DirInfo) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    for (auto &DepInfo : DirEnv.DependsInfo.List) {
      assert(!isa<MultiDependInfo>(DepInfo.get()) && "release directive does not support multideps");
      unpackReleaseDepCallToRT(M, DepInfo.get(), DirInfo.Entry);
    }
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

  // Rewrites task_args using address_translation
  void translateDep(
      IRBuilder<> &IRBTranslate, IRBuilder<> &IRBReload, const DependInfo *DepInfo, Value *DSA,
      Value *&UnpackedDSA, Value *AddrTranslationTable,
      const std::map<Value *, int> &DepSymToIdx) {

    Function *ComputeDepFun = cast<Function>(DepInfo->ComputeDepFun);
    CallInst *CallComputeDep = IRBTranslate.CreateCall(ComputeDepFun, DepInfo->Args);
    llvm::Value *DepBase = IRBTranslate.CreateExtractValue(CallComputeDep, 0);

    Value *Idx[2];
    Idx[0] = ConstantInt::get(Type::getInt32Ty(
      IRBTranslate.getContext()), DepSymToIdx.at(DSA));
    Idx[1] = Constant::getNullValue(Type::getInt32Ty(IRBTranslate.getContext()));
    Value *LocalAddr = IRBTranslate.CreateGEP(
        AddrTranslationTable, Idx, "local_lookup_" + DSA->getName());
    LocalAddr = IRBTranslate.CreateLoad(LocalAddr);

    Idx[1] = ConstantInt::get(Type::getInt32Ty(IRBTranslate.getContext()), 1);
    Value *DeviceAddr = IRBTranslate.CreateGEP(
        AddrTranslationTable, Idx, "device_lookup_" + DSA->getName());
    DeviceAddr = IRBTranslate.CreateLoad(DeviceAddr);

    // Res = device_addr + (DSA_addr - local_addr)
    Value *Translation = IRBTranslate.CreateBitCast(
      DepBase, Type::getInt8PtrTy(IRBTranslate.getContext()));
    Translation = IRBTranslate.CreateGEP(Translation, IRBTranslate.CreateNeg(LocalAddr));
    Translation = IRBTranslate.CreateGEP(Translation, DeviceAddr);

    // Store the translation in task_args
    if (auto *LUnpackedDSA = dyn_cast<LoadInst>(UnpackedDSA)) {
      Translation = IRBTranslate.CreateBitCast(Translation, LUnpackedDSA->getType());
      IRBTranslate.CreateStore(Translation, LUnpackedDSA->getPointerOperand());
      // Reload what we have translated
      UnpackedDSA = IRBReload.CreateLoad(LUnpackedDSA->getPointerOperand());
    } else {
      Translation = IRBTranslate.CreateBitCast(
        Translation, UnpackedDSA->getType()->getPointerElementType());
      IRBTranslate.CreateStore(Translation, UnpackedDSA);
    }
  }

  // Given a Outline Function assuming that task args are the first parameter, and
  // DSAInfo and VLADimsInfo, it unpacks task args in Outline and fills UnpackedList
  // with those Values, used to call Unpack Functions
  void unpackDSAsWithVLADims(Module &M, const DirectiveInfo &DirInfo,
                  Function *OlFunc,
                  const MapVector<Value *, size_t> &StructToIdxMap,
                  SmallVectorImpl<Value *> &UnpackedList) {
    UnpackedList.clear();

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveDSAInfo &DSAInfo = DirEnv.DSAInfo;
    const DirectiveCapturedInfo &CapturedInfo = DirEnv.CapturedInfo;
    const DirectiveVLADimsInfo &VLADimsInfo = DirEnv.VLADimsInfo;

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
  void olCallToUnpack(Module &M, const DirectiveInfo &DirInfo,
                      const MapVector<Value *, size_t> &StructToIdxMap,
                      Function *OlFunc, Function *UnpackFunc,
                      bool IsTaskFunc=false) {
    BasicBlock::Create(M.getContext(), "entry", OlFunc);
    IRBuilder<> BBBuilder(&OlFunc->getEntryBlock());

    // First arg is the nanos_task_args
    Function::arg_iterator AI = OlFunc->arg_begin();
    AI++;
    SmallVector<Value *, 4> UnpackParams;
    unpackDSAsWithVLADims(M, DirInfo, OlFunc, StructToIdxMap, UnpackParams);
    while (AI != OlFunc->arg_end()) {
      UnpackParams.push_back(&*AI++);
    }

    if (IsTaskFunc) {
      // Build call to compute_dep in order to have get the base dependency of
      // the reduction. The result is passed to unpack
      const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
      const DirectiveDependsInfo &DependsInfo = DirEnv.DependsInfo;
      // Preserve the params before translation. And replace used after build all
      // compute_dep calls
      // NOTE: this assumes UnpackParams can be indexed with StructToIdxMap
      SmallVector<Value *, 4> UnpackParamsCopy(UnpackParams);

      BasicBlock *IfThenBB = BasicBlock::Create(M.getContext(), "", OlFunc);
      BasicBlock *IfEndBB = BasicBlock::Create(M.getContext(), "", OlFunc);

      Value *AddrTranslationTable = &*(OlFunc->arg_end() - 1);
      Value *Cmp = BBBuilder.CreateICmpNE(
        AddrTranslationTable, Constant::getNullValue(AddrTranslationTable->getType()));
      BBBuilder.CreateCondBr(Cmp, IfThenBB, IfEndBB);

      IRBuilder<> IRBIfThen(IfThenBB);
      IRBuilder<> IRBIfEnd(IfEndBB);

      for (auto &DepInfo : DependsInfo.List) {
        if (DepInfo->isReduction()) {
          Value *DepBaseDSA = DepInfo->Base;
          translateDep(
            IRBIfThen, IRBIfEnd, DepInfo.get(), DepBaseDSA,
            UnpackParams[StructToIdxMap.lookup(DepBaseDSA)],
            AddrTranslationTable, DirInfo.DirEnv.DepSymToIdx);
        }
      }
      for (Instruction &I : *IfThenBB) {
        auto UnpackedIt = UnpackParamsCopy.begin();
        for (auto It = StructToIdxMap.begin();
               It != StructToIdxMap.end(); ++It, ++UnpackedIt) {
          if (isReplaceableValue(It->first))
            I.replaceUsesOfWith(It->first, *UnpackedIt);
        }
      }
      IRBIfThen.CreateBr(IfEndBB);
      // Build TaskUnpackCall
      IRBIfEnd.CreateCall(UnpackFunc, UnpackParams);
      IRBIfEnd.CreateRetVoid();
    } else {
      // Build TaskUnpackCall
      BBBuilder.CreateCall(UnpackFunc, UnpackParams);
      BBBuilder.CreateRetVoid();
    }
  }

  // Copy task_args from src to dst, calling copyctors or ctors if
  // nonpods
  void duplicateArgs(Module &M, const DirectiveInfo &DirInfo,
                     const MapVector<Value *, size_t> &StructToIdxMap,
                     Function *OlFunc, StructType *TaskArgsTy) {
    BasicBlock::Create(M.getContext(), "entry", OlFunc);
    IRBuilder<> IRB(&OlFunc->getEntryBlock());

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveDSAInfo &DSAInfo = DirEnv.DSAInfo;
    const DirectiveCapturedInfo &CapturedInfo = DirEnv.CapturedInfo;
    const DirectiveVLADimsInfo &VLADimsInfo = DirEnv.VLADimsInfo;
    const DirectiveNonPODsInfo &NonPODsInfo = DirEnv.NonPODsInfo;

    Function::arg_iterator AI = OlFunc->arg_begin();
    Value *TaskArgsSrc = &*AI++;
    Value *TaskArgsDst = &*AI++;
    Value *TaskArgsDstL = IRB.CreateLoad(TaskArgsDst);

    SmallVector<VLAAlign, 2> VLAAlignsInfo;
    computeVLAsAlignOrder(M, VLAAlignsInfo, VLADimsInfo);

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

    Value *TaskArgsDstLi8 = IRB.CreateBitCast(TaskArgsDstL, IRB.getInt8PtrTy());
    Value *TaskArgsDstLi8IdxGEP = IRB.CreateGEP(TaskArgsDstLi8, TaskArgsStructSizeOf, "args_end");

    // First point VLAs to its according space in task args
    for (const VLAAlign& VAlign : VLAAlignsInfo) {
      auto *V = VAlign.V;
      unsigned TyAlign = VAlign.Align;

      Type *Ty = V->getType()->getPointerElementType();

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
      Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(V));
      Value *GEP = IRB.CreateGEP(
          TaskArgsDstL, Idx, "gep_dst_" + V->getName());

      // Point VLA in task args to an aligned position of the extra space allocated
      Value *GEPi8 = IRB.CreateBitCast(GEP, IRB.getInt8PtrTy()->getPointerTo());
      IRB.CreateAlignedStore(TaskArgsDstLi8IdxGEP, GEPi8, Align(TyAlign));
      // Skip current VLA size
      unsigned SizeB = M.getDataLayout().getTypeAllocSize(Ty);
      Value *VLASize = ConstantInt::get(IRB.getInt64Ty(), SizeB);
      for (auto *Dim : VLADimsInfo.lookup(V)) {
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

    for (auto *V : DSAInfo.Shared) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
      Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(V));
      Value *GEPSrc = IRB.CreateGEP(
          TaskArgsSrc, Idx, "gep_src_" + V->getName());
      Value *GEPDst = IRB.CreateGEP(
          TaskArgsDstL, Idx, "gep_dst_" + V->getName());
      IRB.CreateStore(IRB.CreateLoad(GEPSrc), GEPDst);
    }
    for (auto *V : DSAInfo.Private) {
      // Call custom constructor generated in clang in non-pods
      // Leave pods unititialized
      auto It = NonPODsInfo.Inits.find(V);
      if (It != NonPODsInfo.Inits.end()) {
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
        } else if (VLADimsInfo.count(V)) {
          for (auto *Dim : VLADimsInfo.lookup(V)) {
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
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(V));
        Value *GEP = IRB.CreateGEP(
            TaskArgsDstL, Idx, "gep_" + V->getName());

        // VLAs
        if (VLADimsInfo.count(V))
          GEP = IRB.CreateLoad(GEP);

        // Regular arrays have types like [10 x %struct.S]*
        // Cast to %struct.S*
        GEP = IRB.CreateBitCast(GEP, Ty->getPointerTo());

        IRB.CreateCall(FunctionCallee(cast<Function>(It->second)), ArrayRef<Value*>{GEP, NSize});
      }
    }
    for (auto *V : DSAInfo.Firstprivate) {
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
      } else if (VLADimsInfo.count(V)) {
        for (auto *Dim : VLADimsInfo.lookup(V)) {
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
      Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(V));
      Value *GEPSrc = IRB.CreateGEP(
          TaskArgsSrc, Idx, "gep_src_" + V->getName());
      Value *GEPDst = IRB.CreateGEP(
          TaskArgsDstL, Idx, "gep_dst_" + V->getName());

      // VLAs
      if (VLADimsInfo.count(V)) {
        GEPSrc = IRB.CreateLoad(GEPSrc);
        GEPDst = IRB.CreateLoad(GEPDst);
      }

      auto It = NonPODsInfo.Copies.find(V);
      if (It != NonPODsInfo.Copies.end()) {
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
    for (Value *V : CapturedInfo) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
      Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(V));
      Value *GEPSrc = IRB.CreateGEP(
          TaskArgsSrc, Idx, "capt_gep_src_" + V->getName());
      Value *GEPDst = IRB.CreateGEP(
          TaskArgsDstL, Idx, "capt_gep_dst_" + V->getName());
      IRB.CreateStore(IRB.CreateLoad(GEPSrc), GEPDst);
    }

    IRB.CreateRetVoid();
  }

  Value *computeTaskArgsVLAsExtraSizeOf(
      Module &M, IRBuilder<> &IRB, const DirectiveVLADimsInfo &VLADimsInfo) {
    Value *Sum = ConstantInt::get(IRB.getInt64Ty(), 0);
    for (const auto &VLAWithDimsMap : VLADimsInfo) {
      Type *Ty = VLAWithDimsMap.first->getType()->getPointerElementType();
      unsigned SizeB = M.getDataLayout().getTypeAllocSize(Ty);
      Value *ArraySize = ConstantInt::get(IRB.getInt64Ty(), SizeB);
      for (auto *V : VLAWithDimsMap.second) {
        ArraySize = IRB.CreateNUWMul(ArraySize, V);
      }
      Sum = IRB.CreateNUWAdd(Sum, ArraySize);
    }
    return Sum;
  }

  StructType *createTaskArgsType(Module &M,
                                 const DirectiveInfo &DirInfo,
                                 MapVector<Value *, size_t> &StructToIdxMap, StringRef Str) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveDSAInfo &DSAInfo = DirEnv.DSAInfo;
    const DirectiveCapturedInfo &CapturedInfo = DirEnv.CapturedInfo;
    const DirectiveVLADimsInfo &VLADimsInfo = DirEnv.VLADimsInfo;
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
  void computeVLAsAlignOrder(
      Module &M, SmallVectorImpl<VLAAlign> &VLAAlignsInfo, const DirectiveVLADimsInfo &VLADimsInfo) {
    for (const auto &VLAWithDimsMap : VLADimsInfo) {
      auto *V = VLAWithDimsMap.first;
      Type *Ty = V->getType()->getPointerElementType();

      unsigned Align = M.getDataLayout().getPrefTypeAlignment(Ty);

      auto It = VLAAlignsInfo.begin();
      while (It != VLAAlignsInfo.end() && It->Align >= Align)
        ++It;

      VLAAlignsInfo.insert(It, {V, Align});
    }
  }

  void lowerTaskwait(const DirectiveInfo &DirInfo,
                     Module &M) {
    // 1. Create Taskwait function Type
    IRBuilder<> IRB(DirInfo.Entry);
    FunctionCallee Func = M.getOrInsertFunction(
        "nanos6_taskwait", IRB.getVoidTy(), IRB.getInt8PtrTy());
    // 2. Build String
    unsigned Line = DirInfo.Entry->getDebugLoc().getLine();
    unsigned Col = DirInfo.Entry->getDebugLoc().getCol();

    std::string FileNamePlusLoc = (M.getSourceFileName()
                                   + ":" + Twine(Line)
                                   + ":" + Twine(Col)).str();
    Constant *Nanos6TaskwaitLocStr = IRB.CreateGlobalStringPtr(FileNamePlusLoc);

    // 3. Insert the call
    IRB.CreateCall(Func, {Nanos6TaskwaitLocStr});
    // 4. Remove the intrinsic
    DirInfo.Entry->eraseFromParent();
  }

  void lowerRelease(const DirectiveInfo &DirInfo,
                    Module &M) {
    unpackReleaseDepsCallToRT(M, DirInfo);
    // Remove the intrinsic
    DirInfo.Entry->eraseFromParent();
  }

  // This must be called before erasing original entry/exit
  void buildFinalCondCFG(
      Module &M, Function &F, ArrayRef<FinalBodyInfo> FinalInfos) {
    // Lower final inner tasks

    // Skip first since its the current task
    for (size_t i = 1; i < FinalInfos.size(); ++i) {
      // Build loop for taskloop/taskfor
      bool IsLoop = FinalInfos[i].DirInfo->DirEnv.isOmpSsLoopDirective();
      if (IsLoop) {
        const DirectiveLoopInfo &LoopInfo = FinalInfos[i].DirInfo->DirEnv.LoopInfo;
        buildLoopForTask(M, F, FinalInfos[i].CloneEntry, FinalInfos[i].CloneExit, LoopInfo);
      }
      FinalInfos[i].CloneExit->eraseFromParent();
      FinalInfos[i].CloneEntry->eraseFromParent();
    }

    BasicBlock *OrigEntryBB = FinalInfos[0].DirInfo->Entry->getParent();
    BasicBlock *OrigExitBB = FinalInfos[0].DirInfo->Exit->getParent()->getUniqueSuccessor();

    BasicBlock *CloneEntryBB = FinalInfos[0].CloneEntry->getParent();
    bool IsLoop = FinalInfos[0].DirInfo->DirEnv.isOmpSsLoopDirective();
    if (IsLoop) {
      const DirectiveLoopInfo &LoopInfo = FinalInfos[0].DirInfo->DirEnv.LoopInfo;
      CloneEntryBB = buildLoopForTask(M, F, FinalInfos[0].CloneEntry, FinalInfos[0].CloneExit, LoopInfo);
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

  Function *createDestroyArgsOlFunc(
      Module &M, Function &F, int taskNum,
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, const DirectiveInfo &DirInfo,
      ArrayRef<Type *> TaskTypeList, ArrayRef<StringRef> TaskNameList) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveNonPODsInfo &NonPODsInfo = DirEnv.NonPODsInfo;

    // Do not create anything
    if (NonPODsInfo.Deinits.empty())
      return nullptr;

    Function *UnpackDestroyArgsFuncVar
      = createUnpackOlFunction(
        M, F, ("nanos6_unpacked_destroy_" + F.getName() + Twine(taskNum)).str(),
        TaskTypeList, TaskNameList, {}, {});
    unpackDestroyArgsAndRewrite(M, DirInfo, UnpackDestroyArgsFuncVar, TaskArgsToStructIdxMap);

    // nanos6_unpacked_destroy_* END

    // nanos6_ol_destroy_* START

    Function *OlDestroyArgsFuncVar
      = createUnpackOlFunction(
        M, F, ("nanos6_ol_destroy_" + F.getName() + Twine(taskNum)).str(),
        {TaskArgsTy->getPointerTo()}, {"task_args"}, {}, {});
    olCallToUnpack(M, DirInfo, TaskArgsToStructIdxMap, OlDestroyArgsFuncVar, UnpackDestroyArgsFuncVar);

    return OlDestroyArgsFuncVar;

  }

  Function *createDuplicateArgsOlFunc(
      Module &M, Function &F, int taskNum,
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, const DirectiveInfo &DirInfo,
      ArrayRef<Type *> TaskTypeList, ArrayRef<StringRef> TaskNameList) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveNonPODsInfo &NonPODsInfo = DirEnv.NonPODsInfo;
    const DirectiveVLADimsInfo &VLADimsInfo = DirEnv.VLADimsInfo;

    // Do not create anything if all are PODs
    // and there are no VLAs.
    // The runtime will perform a memcpy
    if (NonPODsInfo.Inits.empty() &&
        NonPODsInfo.Copies.empty() &&
        VLADimsInfo.empty())
      return nullptr;

    Function *OlDuplicateArgsFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_ol_duplicate_" + F.getName() + Twine(taskNum)).str(),
                               {TaskArgsTy->getPointerTo(), TaskArgsTy->getPointerTo()->getPointerTo()}, {"task_args_src", "task_args_dst"},
                               {}, {});
    duplicateArgs(M, DirInfo, TaskArgsToStructIdxMap, OlDuplicateArgsFuncVar, TaskArgsTy);

    return OlDuplicateArgsFuncVar;
  }

  void createTaskFuncOl(
      Module &M, Function &F, int taskNum,
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, const DirectiveInfo &DirInfo,
      ArrayRef<Type *> TaskTypeList, ArrayRef<StringRef> TaskNameList,
      bool IsLoop, Function *&OlFunc, Function *&UnpackFunc) {

    SmallVector<Type *, 4> TaskExtraTypeList;
    SmallVector<StringRef, 4> TaskExtraNameList;

    if (IsLoop) {
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

    // CodeExtractor will create a entry block for us
    UnpackFunc = createUnpackOlFunction(
      M, F, ("nanos6_unpacked_task_region_" + F.getName() + Twine(taskNum)).str(),
      TaskTypeList, TaskNameList, TaskExtraTypeList, TaskExtraNameList);

    OlFunc = createUnpackOlFunction(
      M, F, ("nanos6_ol_task_region_" + F.getName() + Twine(taskNum)).str(),
      {TaskArgsTy->getPointerTo()}, {"task_args"}, TaskExtraTypeList, TaskExtraNameList);

    olCallToUnpack(M, DirInfo, TaskArgsToStructIdxMap, OlFunc, UnpackFunc, /*IsTaskFunc=*/true);
  }

  Function *createDepsOlFunc(
      Module &M, Function &F, int taskNum,
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, const DirectiveInfo &DirInfo,
      ArrayRef<Type *> TaskTypeList, ArrayRef<StringRef> TaskNameList) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveDependsInfo &DependsInfo = DirEnv.DependsInfo;

    // Do not do anything if there are no dependencies
    if (DependsInfo.List.empty())
      return nullptr;

    SmallVector<Type *, 4> TaskExtraTypeList;
    SmallVector<StringRef, 4> TaskExtraNameList;

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
    unpackDepsAndRewrite(M, DirInfo, UnpackDepsFuncVar, TaskArgsToStructIdxMap);

    Function *OlDepsFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_ol_deps_" + F.getName() + Twine(taskNum)).str(),
                               {TaskArgsTy->getPointerTo()}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);
    olCallToUnpack(M, DirInfo, TaskArgsToStructIdxMap, OlDepsFuncVar, UnpackDepsFuncVar);

    return OlDepsFuncVar;
  }

  Function *createConstraintsOlFunc(
      Module &M, Function &F, int taskNum,
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, const DirectiveInfo &DirInfo,
      ArrayRef<Type *> TaskTypeList, ArrayRef<StringRef> TaskNameList) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;

    if (!DirEnv.Cost)
      return nullptr;

    SmallVector<Type *, 4> TaskExtraTypeList;
    SmallVector<StringRef, 4> TaskExtraNameList;

    // nanos6_task_constraints_t *constraints
    TaskExtraTypeList.push_back(Nanos6TaskConstraints::getInstance(M).getType()->getPointerTo());
    TaskExtraNameList.push_back("constraints");

    Function *UnpackConstraintsFuncVar = createUnpackOlFunction(M, F,
                               ("nanos6_unpacked_constraints_" + F.getName() + Twine(taskNum)).str(),
                               TaskTypeList, TaskNameList,
                               TaskExtraTypeList, TaskExtraNameList);
    unpackCostAndRewrite(M, DirEnv.Cost, UnpackConstraintsFuncVar, TaskArgsToStructIdxMap);

    Function *OlConstraintsFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_ol_constraints_" + F.getName() + Twine(taskNum)).str(),
                               {TaskArgsTy->getPointerTo()}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);
    olCallToUnpack(M, DirInfo, TaskArgsToStructIdxMap, OlConstraintsFuncVar, UnpackConstraintsFuncVar);

    return OlConstraintsFuncVar;
  }

  Function *createPriorityOlFunc(
      Module &M, Function &F, int taskNum,
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, const DirectiveInfo &DirInfo,
      ArrayRef<Type *> TaskTypeList, ArrayRef<StringRef> TaskNameList) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;

    if (!DirEnv.Priority)
      return nullptr;

    SmallVector<Type *, 4> TaskExtraTypeList;
    SmallVector<StringRef, 4> TaskExtraNameList;

    // nanos6_priority_t *priority
    // long int *priority
    TaskExtraTypeList.push_back(Type::getInt64Ty(M.getContext())->getPointerTo());
    TaskExtraNameList.push_back("priority");

    Function *UnpackPriorityFuncVar = createUnpackOlFunction(M, F,
                               ("nanos6_unpacked_priority_" + F.getName() + Twine(taskNum)).str(),
                               TaskTypeList, TaskNameList,
                               TaskExtraTypeList, TaskExtraNameList);
    unpackPriorityAndRewrite(M, DirEnv.Priority, UnpackPriorityFuncVar, TaskArgsToStructIdxMap);

    Function *OlPriorityFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_ol_priority_" + F.getName() + Twine(taskNum)).str(),
                               {TaskArgsTy->getPointerTo()}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);
    olCallToUnpack(M, DirInfo, TaskArgsToStructIdxMap, OlPriorityFuncVar, UnpackPriorityFuncVar);

    return OlPriorityFuncVar;
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
  //         //! Specifies that the task has the "wait" clause
  //         nanos6_waiting_task = (1 << 4),
  //         //! Specifies that the args_block is preallocated from user side
  //         nanos6_preallocated_args_block = (1 << 5),
  //         //! Specifies that the task has been verified by the user, hence it doesn't need runtime linting
  //         nanos6_verified_task = (1 << 6)
  // } nanos6_task_flag_t;
  Value *computeTaskFlags(IRBuilder<> &IRB, const DirectiveEnvironment &DirEnv) {
    Value *TaskFlagsVar = ConstantInt::get(IRB.getInt64Ty(), 0);
    if (DirEnv.Final) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateZExt(DirEnv.Final,
                         IRB.getInt64Ty()));
    }
    if (DirEnv.If) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
            IRB.CreateZExt(
              IRB.CreateICmpEQ(DirEnv.If, IRB.getFalse()),
              IRB.getInt64Ty()),
              1));
    }
    if (DirEnv.isOmpSsTaskLoopDirective()) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
              ConstantInt::get(IRB.getInt64Ty(), 1),
              2));
    }
    if (DirEnv.isOmpSsTaskForDirective()) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
              ConstantInt::get(IRB.getInt64Ty(), 1),
              3));
    }
    if (DirEnv.Wait) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
            IRB.CreateZExt(
              DirEnv.Wait,
              IRB.getInt64Ty()),
              4));
    }
    return TaskFlagsVar;
  }

  void lowerTask(const DirectiveInfo &DirInfo,
                 Function &F,
                 size_t taskNum,
                 Module &M,
                 FinalBodyInfoMap &FinalBodyInfo) {

    DebugLoc DLoc = DirInfo.Entry->getDebugLoc();
    unsigned Line = DLoc.getLine();
    unsigned Col = DLoc.getCol();
    std::string FileNamePlusLoc = (M.getSourceFileName()
                                   + ":" + Twine(Line)
                                   + ":" + Twine(Col)).str();

    Constant *Nanos6TaskLocStr = IRBuilder<>(DirInfo.Entry).CreateGlobalStringPtr(FileNamePlusLoc);

    buildFinalCondCFG(M, F, FinalBodyInfo.lookup(&DirInfo));

    BasicBlock *EntryBB = DirInfo.Entry->getParent();
    BasicBlock *ExitBB = DirInfo.Exit->getParent()->getUniqueSuccessor();

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveLoopInfo LoopInfo = DirInfo.DirEnv.LoopInfo;

    // In loop constructs this will be the starting loop BB
    BasicBlock *NewEntryBB = EntryBB;
    DirectiveLoopInfo NewLoopInfo = LoopInfo;
    if (!LoopInfo.empty()) {
      Type *IndVarTy = LoopInfo.IndVar->getType()->getPointerElementType();

      IRBuilder<> IRB(DirInfo.Entry);

      // Use tmp variables to be replaced by what comes from nanos6. This fixes
      // the problem when bounds or step are constants
      NewLoopInfo.LBound = IRB.CreateAlloca(IndVarTy, nullptr, "lb.tmp.addr");
      IRB.CreateStore(createZSExtOrTrunc(IRB, LoopInfo.LBound, IndVarTy, LoopInfo.LBoundSigned), NewLoopInfo.LBound);
      NewLoopInfo.LBound = IRB.CreateLoad(NewLoopInfo.LBound);
      NewLoopInfo.LBoundSigned = LoopInfo.IndVarSigned;

      NewLoopInfo.UBound = IRB.CreateAlloca(IndVarTy, nullptr, "ub.tmp.addr");
      IRB.CreateStore(createZSExtOrTrunc(IRB, LoopInfo.UBound, IndVarTy, LoopInfo.UBoundSigned), NewLoopInfo.UBound);
      NewLoopInfo.UBound = IRB.CreateLoad(NewLoopInfo.UBound);
      NewLoopInfo.UBoundSigned = LoopInfo.IndVarSigned;

      // unpacked_task_region loops are always step 1
      NewLoopInfo.Step = IRB.CreateAlloca(IndVarTy, nullptr, "step.tmp.addr");
      IRB.CreateStore(ConstantInt::get(IndVarTy, 1), NewLoopInfo.Step);
      NewLoopInfo.Step = IRB.CreateLoad(NewLoopInfo.Step);
      NewLoopInfo.StepSigned = LoopInfo.IndVarSigned;

      NewLoopInfo.IndVar =
        IRB.CreateAlloca(IndVarTy, nullptr, "loop." + LoopInfo.IndVar->getName());
      // unpacked_task_region loops are always SLT
      NewLoopInfo.LoopType = DirectiveLoopInfo::LT;
      buildLoopForTaskImpl(M, F, DirInfo.Entry, DirInfo.Exit, NewLoopInfo, NewEntryBB, EntryBB);
    }

    DirInfo.Exit->eraseFromParent();
    DirInfo.Entry->eraseFromParent();

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
    StructType *TaskArgsTy = createTaskArgsType(M, DirInfo, TaskArgsToStructIdxMap,
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

    Function *OlDestroyArgsFuncVar =
      createDestroyArgsOlFunc(M, F, taskNum, TaskArgsToStructIdxMap, TaskArgsTy, DirInfo, TaskTypeList, TaskNameList);

    Function *OlDuplicateArgsFuncVar =
      createDuplicateArgsOlFunc(M, F, taskNum, TaskArgsToStructIdxMap, TaskArgsTy, DirInfo, TaskTypeList, TaskNameList);

    Function *OlTaskFuncVar = nullptr;
    Function *UnpackTaskFuncVar = nullptr;
      createTaskFuncOl(
        M, F, taskNum, TaskArgsToStructIdxMap, TaskArgsTy, DirInfo,
        TaskTypeList, TaskNameList, DirEnv.isOmpSsLoopDirective(), OlTaskFuncVar, UnpackTaskFuncVar);

    Function *OlDepsFuncVar =
      createDepsOlFunc(M, F, taskNum, TaskArgsToStructIdxMap, TaskArgsTy, DirInfo, TaskTypeList, TaskNameList);

    Function *OlConstraintsFuncVar =
      createConstraintsOlFunc(M, F, taskNum, TaskArgsToStructIdxMap, TaskArgsTy, DirInfo, TaskTypeList, TaskNameList);

    Function *OlPriorityFuncVar =
      createPriorityOlFunc(M, F, taskNum, TaskArgsToStructIdxMap, TaskArgsTy, DirInfo, TaskTypeList, TaskNameList);

    // 3. Create Nanos6 task data structures info
    GlobalVariable *TaskInvInfoVar =
      cast<GlobalVariable>(
        M.getOrInsertGlobal(
          ("task_invocation_info_" + F.getName() + Twine(taskNum)).str(),
          Nanos6TaskInvInfo::getInstance(M).getType()));
    TaskInvInfoVar->setLinkage(GlobalVariable::InternalLinkage);
    TaskInvInfoVar->setAlignment(Align(64));
    TaskInvInfoVar->setConstant(true);
    TaskInvInfoVar->setInitializer(
      ConstantStruct::get(Nanos6TaskInvInfo::getInstance(M).getType(), Nanos6TaskLocStr));

    bool IsConstLabelOrNull = !DirEnv.Label || isa<Constant>(DirEnv.Label);

    GlobalVariable *TaskImplInfoVar =
      cast<GlobalVariable>(M.getOrInsertGlobal(
        ("implementations_var_" + F.getName() + Twine(taskNum)).str(),
        ArrayType::get(Nanos6TaskImplInfo::getInstance(M).getType(), 1)));
    TaskImplInfoVar->setLinkage(GlobalVariable::InternalLinkage);
    TaskImplInfoVar->setAlignment(Align(64));
    TaskImplInfoVar->setConstant(IsConstLabelOrNull);
    TaskImplInfoVar->setInitializer(
      ConstantArray::get(ArrayType::get(Nanos6TaskImplInfo::getInstance(M).getType(), 1), // TODO: More than one implementations?
        ConstantStruct::get(Nanos6TaskImplInfo::getInstance(M).getType(),
        ConstantInt::get(Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(0), 0),
        ConstantExpr::getPointerCast(OlTaskFuncVar, Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(1)),
        OlConstraintsFuncVar
          ? ConstantExpr::getPointerCast(OlConstraintsFuncVar,
                                         Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(2))
          : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(2))),
        DirEnv.Label && isa<Constant>(DirEnv.Label)
          ? ConstantExpr::getPointerCast(cast<Constant>(DirEnv.Label),
                                         Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(3))
          : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(3))),
        Nanos6TaskLocStr,
        ConstantPointerNull::get(cast<PointerType>(Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(5))))));


    GlobalVariable *TaskRedInitsVar =
      cast<GlobalVariable>(
        M.getOrInsertGlobal(("nanos6_reduction_initializers_" + F.getName() + Twine(taskNum)).str(),
        ArrayType::get(
          FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo(),
          DirEnv.ReductionsInitCombInfo.size())));
    TaskRedInitsVar->setLinkage(GlobalVariable::InternalLinkage);
    TaskRedInitsVar->setAlignment(Align(64));
    TaskRedInitsVar->setConstant(true);
    {
      SmallVector<Constant *, 4> Inits;
      for (auto &p : DirEnv.ReductionsInitCombInfo) {
        Inits.push_back(
          ConstantExpr::getPointerCast(cast<Constant>(p.second.Init),
          FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo()));
      }
      TaskRedInitsVar->setInitializer(
        ConstantArray::get(ArrayType::get(
          FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo(),
          DirEnv.ReductionsInitCombInfo.size()), Inits));
    }

    GlobalVariable *TaskRedCombsVar =
      cast<GlobalVariable>(
        M.getOrInsertGlobal(("nanos6_reduction_combiners_" + F.getName() + Twine(taskNum)).str(),
        ArrayType::get(
          FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo(),
          DirEnv.ReductionsInitCombInfo.size())));
    TaskRedCombsVar->setLinkage(GlobalVariable::InternalLinkage);
    TaskRedCombsVar->setAlignment(Align(64));
    TaskRedCombsVar->setConstant(true);
    {
      SmallVector<Constant *, 4> Combs;
      for (auto &p : DirEnv.ReductionsInitCombInfo) {
        Combs.push_back(
          ConstantExpr::getPointerCast(cast<Constant>(p.second.Comb),
          FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo()));
      }
      TaskRedCombsVar->setInitializer(
        ConstantArray::get(ArrayType::get(
          FunctionType::get(Type::getVoidTy(M.getContext()), /*IsVarArgs=*/false)->getPointerTo(),
          DirEnv.ReductionsInitCombInfo.size()), Combs));
    }

    GlobalVariable *TaskInfoVar =
      cast<GlobalVariable>(M.getOrInsertGlobal(("task_info_var_" + F.getName() + Twine(taskNum)).str(),
                                      Nanos6TaskInfo::getInstance(M).getType()));
    TaskInfoVar->setLinkage(GlobalVariable::InternalLinkage);
    TaskInfoVar->setAlignment(Align(64));
    TaskInfoVar->setConstant(false); // TaskInfo is modified by nanos6
    TaskInfoVar->setInitializer(
      ConstantStruct::get(Nanos6TaskInfo::getInstance(M).getType(),
      // TODO: Add support for devices
      ConstantInt::get(Nanos6TaskInfo::getInstance(M).getType()->getElementType(0), DirEnv.DependsInfo.NumSymbols),
      OlDepsFuncVar
        ? ConstantExpr::getPointerCast(OlDepsFuncVar,
                                       Nanos6TaskInfo::getInstance(M).getType()->getElementType(1))
        : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(1))),
      OlPriorityFuncVar
        ? ConstantExpr::getPointerCast(OlPriorityFuncVar,
                                       Nanos6TaskInfo::getInstance(M).getType()->getElementType(2))
        : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(2))),
      ConstantInt::get(Nanos6TaskInfo::getInstance(M).getType()->getElementType(3), 1),
      ConstantExpr::getPointerCast(TaskImplInfoVar, Nanos6TaskInfo::getInstance(M).getType()->getElementType(4)),
      OlDestroyArgsFuncVar
        ? ConstantExpr::getPointerCast(OlDestroyArgsFuncVar,
                                       Nanos6TaskInfo::getInstance(M).getType()->getElementType(5))
        : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(5))),
      OlDuplicateArgsFuncVar
        ? ConstantExpr::getPointerCast(OlDuplicateArgsFuncVar,
                                       Nanos6TaskInfo::getInstance(M).getType()->getElementType(6))
        : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(6))),
      ConstantExpr::getPointerCast(TaskRedInitsVar, cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(7))),
      ConstantExpr::getPointerCast(TaskRedCombsVar, cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(8))),
      ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(9)))));

    registerTaskInfo(M, TaskInfoVar);

    auto rewriteUsesBrAndGetOmpSsUnpackFunc
      = [&M, &LoopInfo, &EntryBB, &NewLoopInfo, &UnpackTaskFuncVar, &TaskArgsToStructIdxMap](BasicBlock *header,
                                            BasicBlock *newRootNode,
                                            BasicBlock *newHeader,
                                            Function *oldFunction,
                                            const SetVector<BasicBlock *> &Blocks) {
      UnpackTaskFuncVar->getBasicBlockList().push_back(newRootNode);

      if (!LoopInfo.empty()) {
        Type *IndVarTy = LoopInfo.IndVar->getType()->getPointerElementType();

        IRBuilder<> IRB(&header->front());
        Value *LoopBounds = &*(UnpackTaskFuncVar->arg_end() - 2);

        Value *Idx[2];
        Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
        Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
        Value *LBoundField = IRB.CreateGEP(LoopBounds, Idx, "lb_gep");
        LBoundField = IRB.CreateLoad(LBoundField);
        LBoundField = IRB.CreateZExtOrTrunc(LBoundField, IndVarTy, "lb");

        Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 1);
        Value *UBoundField = IRB.CreateGEP(LoopBounds, Idx, "ub_gep");
        UBoundField = IRB.CreateLoad(UBoundField);
        UBoundField = IRB.CreateZExtOrTrunc(UBoundField, IndVarTy, "ub");

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
        auto p = buildInstructionSignDependent(
          LoopBodyIRB, M, NormVal, LoopInfo.Step, NewLoopInfo.IndVarSigned, LoopInfo.StepSigned,
          [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
            return IRB.CreateMul(LHS, RHS);
          });
        p = buildInstructionSignDependent(
          LoopBodyIRB, M, p.first, LoopInfo.LBound, p.second, LoopInfo.LBoundSigned,
          [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
            return IRB.CreateAdd(LHS, RHS);
          });
        LoopBodyIRB.CreateStore(createZSExtOrTrunc(LoopBodyIRB, p.first, IndVarTy, p.second), LoopInfo.IndVar);
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
         &DirEnv, &TaskArgsToStructIdxMap,
         &TaskInfoVar, &TaskImplInfoVar,
         &TaskInvInfoVar](Function *newFunction,
                          BasicBlock *codeReplacer,
                          const SetVector<BasicBlock *> &Blocks) {

      const DirectiveDSAInfo &DSAInfo = DirEnv.DSAInfo;
      const DirectiveVLADimsInfo &VLADimsInfo = DirEnv.VLADimsInfo;
      const DirectiveCapturedInfo &CapturedInfo = DirEnv.CapturedInfo;
      const DirectiveDependsInfo &DependsInfo = DirEnv.DependsInfo;

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

            NewRetBB = BasicBlock::Create(M.getContext(), ".exitStub", newFunction);
            IRBuilder<> (NewRetBB).CreateRetVoid();

            // rewrite the original branch instruction with this new target
            DirInfo->setSuccessor(i, NewRetBB);
          }
      }

      // Here we have a valid codeReplacer BasicBlock with its terminator
      IRB.SetInsertPoint(codeReplacer->getTerminator());

      AllocaInst *TaskArgsVar = IRB.CreateAlloca(TaskArgsTy->getPointerTo());
      Value *TaskArgsVarCast = IRB.CreateBitCast(TaskArgsVar, IRB.getInt8PtrTy()->getPointerTo());
      Value *TaskFlagsVar = computeTaskFlags(IRB, DirEnv);
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

      Value *TaskArgsVLAsExtraSizeOf = computeTaskArgsVLAsExtraSizeOf(M, IRB, VLADimsInfo);
      Value *TaskArgsSizeOf = IRB.CreateNUWAdd(TaskArgsStructSizeOf, TaskArgsVLAsExtraSizeOf);

      Instruction *NumDependencies = IRB.CreateAlloca(IRB.getInt64Ty(), nullptr, "num.deps");
      if (DirEnv.isOmpSsTaskLoopDirective()) {
        // If taskloop NumDeps = -1
        IRB.CreateStore(IRB.getInt64(-1), NumDependencies);
      } else {
        IRB.CreateStore(IRB.getInt64(0), NumDependencies);
        for (auto &DepInfo : DependsInfo.List) {
          Instruction *NumDependenciesLoad = IRB.CreateLoad(NumDependencies);
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

            Function *ComputeMultiDepFun = MultiDepInfo->ComputeMultiDepFun;
            auto Args = MultiDepInfo->Args;
            for (size_t i = 0; i < MultiDepInfo->Iters.size(); i++) {
              Value *IndVar = MultiDepInfo->Iters[i];
              auto LBoundGen = [ComputeMultiDepFun, &Args, i](IRBuilder<> &IRB) {
                Value *ComputeMultiDepCall = IRB.CreateCall(ComputeMultiDepFun, Args);
                 return IRB.CreateExtractValue(ComputeMultiDepCall, i*(3 + 1) + 0);
              };
              auto RemapGen = [IndVar](IRBuilder<> &IRB) {
                // We do not need remap here
                return IRB.CreateLoad(IndVar);
              };
              auto UBoundGen = [ComputeMultiDepFun, &Args, i](IRBuilder<> &IRB) {
                Value *ComputeMultiDepCall = IRB.CreateCall(ComputeMultiDepFun, Args);
                 return IRB.CreateExtractValue(ComputeMultiDepCall, i*(3 + 1) + 2);
              };
              auto IncrGen = [ComputeMultiDepFun, &Args, i](IRBuilder<> &IRB) {
                Value *ComputeMultiDepCall = IRB.CreateCall(ComputeMultiDepFun, Args);
                 return IRB.CreateExtractValue(ComputeMultiDepCall, i*(3 + 1) + 3);
              };
              buildLoopForMultiDep(
                  M, F, NumDependenciesLoad, NumDependenciesStore, IndVar, LBoundGen, RemapGen, UBoundGen, IncrGen);
            }
          }
        }
      }

      // Store label if it's not a string literal (i.e label("L1"))
      if (DirEnv.Label && !isa<Constant>(DirEnv.Label)) {
        Value *Idx[3];
        Idx[0] = Constant::getNullValue(Type::getInt32Ty(IRB.getContext()));
        Idx[1] = Constant::getNullValue(Type::getInt32Ty(IRB.getContext()));
        Idx[2] = ConstantInt::get(Type::getInt32Ty(IRB.getContext()), 3);
        Value *LabelField = IRB.CreateGEP(TaskImplInfoVar, Idx, "ASDF");
        IRB.CreateStore(DirEnv.Label, LabelField);
      }

      IRB.CreateCall(CreateTaskFuncCallee, {TaskInfoVar,
                                  TaskInvInfoVar,
                                  TaskArgsSizeOf,
                                  TaskArgsVarCast,
                                  TaskPtrVar,
                                  TaskFlagsVar,
                                  IRB.CreateLoad(NumDependencies)});

      // DSA capture
      Value *TaskArgsVarL = IRB.CreateLoad(TaskArgsVar);

      Value *TaskArgsVarLi8 = IRB.CreateBitCast(TaskArgsVarL, IRB.getInt8PtrTy());
      Value *TaskArgsVarLi8IdxGEP = IRB.CreateGEP(TaskArgsVarLi8, TaskArgsStructSizeOf, "args_end");

      SmallVector<VLAAlign, 2> VLAAlignsInfo;
      computeVLAsAlignOrder(M, VLAAlignsInfo, VLADimsInfo);

      // First point VLAs to its according space in task args
      for (const auto& VAlign : VLAAlignsInfo) {
        auto *V = VAlign.V;
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
        for (auto *Dim : VLADimsInfo.lookup(V))
          VLASize = IRB.CreateNUWMul(VLASize, Dim);
        TaskArgsVarLi8IdxGEP = IRB.CreateGEP(TaskArgsVarLi8IdxGEP, VLASize);
      }

      for (Value *V : DSAInfo.Shared) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsToStructIdxMap[V]);
        Value *GEP = IRB.CreateGEP(
            TaskArgsVarL, Idx, "gep_" + V->getName());
        IRB.CreateStore(V, GEP);
      }
      for (Value *V : DSAInfo.Private) {
        // Call custom constructor generated in clang in non-pods
        // Leave pods unititialized
        auto It = DirEnv.NonPODsInfo.Inits.find(V);
        if (It != DirEnv.NonPODsInfo.Inits.end()) {
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
          } else if (VLADimsInfo.count(V)) {
            for (auto *Dim : VLADimsInfo.lookup(V))
              NSize = IRB.CreateNUWMul(NSize, Dim);
          }

          Value *Idx[2];
          Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
          Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsToStructIdxMap[V]);
          Value *GEP = IRB.CreateGEP(
              TaskArgsVarL, Idx, "gep_" + V->getName());

          // VLAs
          if (VLADimsInfo.count(V))
            GEP = IRB.CreateLoad(GEP);

          // Regular arrays have types like [10 x %struct.S]*
          // Cast to %struct.S*
          GEP = IRB.CreateBitCast(GEP, Ty->getPointerTo());

          IRB.CreateCall(FunctionCallee(cast<Function>(It->second)), ArrayRef<Value*>{GEP, NSize});
        }
      }
      for (Value *V : DSAInfo.Firstprivate) {
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
        } else if (VLADimsInfo.count(V)) {
          for (auto *Dim : VLADimsInfo.lookup(V))
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
        if (VLADimsInfo.count(V))
          GEP = IRB.CreateLoad(GEP);

        auto It = DirEnv.NonPODsInfo.Copies.find(V);
        if (It != DirEnv.NonPODsInfo.Copies.end()) {
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
      for (Value *V : CapturedInfo) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsToStructIdxMap[V]);
        Value *GEP = IRB.CreateGEP(
            TaskArgsVarL, Idx, "capt_gep_" + V->getName());
        IRB.CreateStore(V, GEP);
      }

      Value *TaskPtrVarL = IRB.CreateLoad(TaskPtrVar);

      if (DirEnv.isOmpSsLoopDirective()) {
        // <      0, (ub - 1 - lb) / step + 1
        // <=     0, (ub - lb)     / step + 1
        // >      0, (ub + 1 - lb) / step + 1
        // >=     0, (ub - lb)     / step + 1
        Type *IndVarTy = DirEnv.LoopInfo.IndVar->getType()->getPointerElementType();
        Value *RegisterLowerB = ConstantInt::get(IndVarTy, 0);

        auto p = buildInstructionSignDependent(
          IRB, M, DirEnv.LoopInfo.UBound, DirEnv.LoopInfo.LBound, DirEnv.LoopInfo.UBoundSigned, DirEnv.LoopInfo.LBoundSigned,
          [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
            return IRB.CreateSub(LHS, RHS);
          });
        Value *RegisterUpperB = p.first;

        Value *RegisterGrainsize = ConstantInt::get(IndVarTy, 0);
        if (DirEnv.LoopInfo.Grainsize)
          RegisterGrainsize = DirEnv.LoopInfo.Grainsize;
        Value *RegisterChunksize = ConstantInt::get(IndVarTy, 0);
        if (DirEnv.LoopInfo.Chunksize)
          RegisterChunksize = DirEnv.LoopInfo.Chunksize;

        switch (DirEnv.LoopInfo.LoopType) {
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
        p = buildInstructionSignDependent(
          IRB, M, RegisterUpperB, DirEnv.LoopInfo.Step, p.second, DirEnv.LoopInfo.Step,
          [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
            if (NewOpSigned)
              return IRB.CreateSDiv(LHS, RHS);
            return IRB.CreateUDiv(LHS, RHS);
          });
        RegisterUpperB = IRB.CreateAdd(p.first, ConstantInt::get(p.first->getType(), 1));
        IRB.CreateCall(
          RegisterLoopFuncCallee,
          {
            TaskPtrVarL,
            createZSExtOrTrunc(
              IRB, RegisterLowerB,
              Nanos6LoopBounds::getInstance(M).getType()->getElementType(0), /*Signed=*/false),
            createZSExtOrTrunc(
              IRB, RegisterUpperB,
              Nanos6LoopBounds::getInstance(M).getType()->getElementType(1), p.second),
            createZSExtOrTrunc(
              IRB, RegisterGrainsize,
              Nanos6LoopBounds::getInstance(M).getType()->getElementType(2), /*Signed=*/false),
            createZSExtOrTrunc(
              IRB, RegisterChunksize,
              Nanos6LoopBounds::getInstance(M).getType()->getElementType(3), /*Signed=*/false)
          }
        );
      }

      CallInst *TaskSubmitFuncCall = IRB.CreateCall(TaskSubmitFuncCallee, TaskPtrVarL);
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
      DirectiveFunctionInfo &DirectiveFuncInfo = getAnalysis<OmpSsRegionAnalysisPass>(*F).getFuncInfo();

      FinalBodyInfoMap FinalBodyInfo;

      // First sweep to clone BBs
      for (DirectiveInfo *pDirInfo : DirectiveFuncInfo.PostOrder) {
        DirectiveInfo &DirInfo = *pDirInfo;

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

        Instruction *OrigEntry = DirInfo.Entry;
        Instruction *OrigExit = DirInfo.Exit;
        Instruction *CloneEntry = cast<Instruction>(VMap[OrigEntry]);
        Instruction *CloneExit = cast<Instruction>(VMap[OrigExit]);
        FinalBodyInfo[&DirInfo].push_back({CloneEntry, CloneExit, &DirInfo});
        for (DirectiveInfo *InnerTI : DirInfo.InnerDirectiveInfos) {
          OrigEntry = InnerTI->Entry;
          OrigExit = InnerTI->Exit;
          CloneEntry = cast<Instruction>(VMap[OrigEntry]);
          CloneExit = cast<Instruction>(VMap[OrigExit]);
          FinalBodyInfo[&DirInfo].push_back({CloneEntry, CloneExit, &DirInfo});
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

      size_t taskNum = 0;
      for (DirectiveInfo *pDirInfo : DirectiveFuncInfo.PostOrder) {
        DirectiveInfo &DirInfo = *pDirInfo;
        const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
        if (DirEnv.isOmpSsTaskwaitDirective())
          lowerTaskwait(DirInfo, M);
        else if (DirEnv.isOmpSsReleaseDirective())
          lowerRelease(DirInfo, M);
        else if (DirEnv.isOmpSsTaskDirective())
          lowerTask(DirInfo, *F, taskNum++, M, FinalBodyInfo);
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
