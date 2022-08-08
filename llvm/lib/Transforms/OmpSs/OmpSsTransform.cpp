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
#include "llvm/IR/DIBuilder.h"
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

struct OmpSs {
  OmpSs(
    function_ref<OmpSsRegionAnalysis &(Function &)> LookupDirectiveFunctionInfo)
      : LookupDirectiveFunctionInfo(LookupDirectiveFunctionInfo)
        {}
  function_ref<OmpSsRegionAnalysis &(Function &)> LookupDirectiveFunctionInfo;

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
        Type *InvSourceTy = PointerType::getUnqual(M.getContext());

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
        Type *RunFuncTy = PointerType::getUnqual(M.getContext());
        // void (*get_constraints)(void *, nanos6_task_constraints_t *);
        Type *GetConstraintsFuncTy = PointerType::getUnqual(M.getContext());
        // const char *task_label;
        Type *TaskLabelTy = PointerType::getUnqual(M.getContext());
        // const char *declaration_source;
        Type *DeclSourceTy = PointerType::getUnqual(M.getContext());
        // const char *device_function_name;
        Type *DevFuncTy = PointerType::getUnqual(M.getContext());
        instance->Ty->setBody({DeviceTypeIdTy, RunFuncTy,
                              GetConstraintsFuncTy, TaskLabelTy,
                              DeclSourceTy, DevFuncTy});
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
        Type *RegisterInfoFuncTy = PointerType::getUnqual(M.getContext());
        // void (*onready_action)(void *args_block);
        Type *OnreadyActionFuncTy = PointerType::getUnqual(M.getContext());
        // void (*get_priority)(void *, nanos6_priority_t *);
        // void (*get_priority)(void *, long int *);
        Type *GetPriorityFuncTy = PointerType::getUnqual(M.getContext());
        // int implementation_count;
        Type *ImplCountTy = Type::getInt32Ty(M.getContext());
        // nanos6_task_implementation_info_t *implementations;
        Type *TaskImplInfoTy = PointerType::getUnqual(M.getContext());
        // void (*destroy_args_block)(void *);
        Type *DestroyArgsBlockFuncTy = PointerType::getUnqual(M.getContext());
        // void (*duplicate_args_block)(const void *, void **);
        Type *DuplicateArgsBlockFuncTy = PointerType::getUnqual(M.getContext());
        // void (**reduction_initializers)(void *, void *, size_t);
        Type *ReductInitsFuncTy = PointerType::getUnqual(M.getContext());
        // void (**reduction_combiners)(void *, void *, size_t);
        Type *ReductCombsFuncTy = PointerType::getUnqual(M.getContext());
        // void *task_type_data;
        Type *TaskTypeDataTy = PointerType::getUnqual(M.getContext());
        // void (*iter_condition)(void *, uint8_t *);
        Type *IterConditionFuncTy = PointerType::getUnqual(M.getContext());
        // int num_args;
        Type *NumArgsTy = Type::getInt32Ty(M.getContext());
        // int *sizeof_table;
        Type *SizeofTableDataTy = PointerType::getUnqual(M.getContext());
        // int *offset_table;
        Type *OffsetTableDataTy = PointerType::getUnqual(M.getContext());
        // int *arg_idx_table;
        Type *ArgIdxTableDataTy = PointerType::getUnqual(M.getContext());

        instance->Ty->setBody({NumSymbolsTy, RegisterInfoFuncTy, OnreadyActionFuncTy, GetPriorityFuncTy,
                               ImplCountTy, TaskImplInfoTy, DestroyArgsBlockFuncTy,
                               DuplicateArgsBlockFuncTy, ReductInitsFuncTy, ReductCombsFuncTy,
                               TaskTypeDataTy, IterConditionFuncTy, NumArgsTy,
                               SizeofTableDataTy, OffsetTableDataTy,
                               ArgIdxTableDataTy
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
        PointerType::getUnqual(M.getContext()),
        Type::getInt32Ty(M.getContext()),
        PointerType::getUnqual(M.getContext()),
        PointerType::getUnqual(M.getContext())
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
      Params.push_back(PointerType::getUnqual(M.getContext()));
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

  // 6 for ndrange and 1 for shm_size
  const size_t DeviceArgsSize = 7;

  // Four each iterator compute the normalized bounds [0, ub) handling
  // outer iterator usage in the current one
  void ComputeLoopBounds(Module &M, const DirectiveLoopInfo &LoopInfo, Instruction *InsertPt,
                         SmallVectorImpl<Value *> &UBounds) {
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
      auto p = buildInstructionSignDependent(
        IRB, M, LoopInfo.UBound[i].Result, LoopInfo.LBound[i].Result, LoopInfo.UBoundSigned[i], LoopInfo.LBoundSigned[i],
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          return IRB.CreateSub(LHS, RHS);
        });
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
      p = buildInstructionSignDependent(
        IRB, M, RegisterUpperB, LoopInfo.Step[i].Result, p.second, LoopInfo.Step[i].Result,
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          if (NewOpSigned)
            return IRB.CreateSDiv(LHS, RHS);
          return IRB.CreateUDiv(LHS, RHS);
        });
      RegisterUpperB = IRB.CreateAdd(p.first, ConstantInt::get(p.first->getType(), 1));

      RegisterUpperB =
        createZSExtOrTrunc(
          IRB, RegisterUpperB,
          Nanos6LoopBounds::getInstance(M).getType()->getElementType(1), p.second);

      UBounds[i] = RegisterUpperB;
    }
  }

  Nanos6MultidepFactory MultidepFactory;

  FunctionCallee CreateTaskFuncCallee;
  FunctionCallee TaskSubmitFuncCallee;
  FunctionCallee CreateLoopFuncCallee;
  FunctionCallee CreateIterFuncCallee;
  FunctionCallee TaskInFinalFuncCallee;
  FunctionCallee TaskInfoRegisterFuncCallee;
  FunctionCallee TaskInfoRegisterCtorFuncCallee;
  FunctionCallee RegisterAssertFuncCallee;
  FunctionCallee RegisterCtorAssertFuncCallee;

  // This is a hack to move some instructions to its corresponding functions
  // at the end of the pass.
  // For example, this is used to move allocas to its corresponding
  // function entry.
  SmallVector<Instruction *, 4> PostMoveInstructions;

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

  // Converts all ConstantExpr users of GV in Blocks to instructions
  static void constantExprToInstruction(
      GlobalValue *GV, const SetVector<BasicBlock *> &Blocks) {
    SmallVector<ConstantExpr*,4> UsersStack;
    SmallVector<Constant*,4> Worklist;
    Worklist.push_back(GV);
    while (!Worklist.empty()) {
      Constant *C = Worklist.pop_back_val();
      for (auto *U : C->users()) {
        if (ConstantExpr *CC = dyn_cast<ConstantExpr>(U)) {
          UsersStack.insert(UsersStack.begin(), CC);
          Worklist.push_back(CC);
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
            Instruction *NewU = U->getAsInstruction();
            NewU->insertBefore(UI);
            UI->replaceUsesOfWith(U, NewU);
          }
        }
      }
    }
  }

  // Build a loop containing Single Entry Single Exit region Entry/Exit
  // and returns the LoopEntry and LoopExit
  void buildLoopForTaskImpl(Module &M, Function &F, Type *IndVarTy, Instruction *Entry,
                        Instruction *Exit, const DirectiveLoopInfo &LoopInfo,
                        Instruction *&LoopEntryI, Instruction *&LoopExitI,
                        SmallVectorImpl<Instruction *> &CollapseStuff,
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
    switch (LoopInfo.LoopType[LInfoIndex]) {
    case DirectiveLoopInfo::LT:
      LoopCmp = buildInstructionSignDependent(
        IRB, M, IndVarVal, LoopInfo.UBound[LInfoIndex].Result, LoopInfo.IndVarSigned[LInfoIndex], LoopInfo.UBoundSigned[LInfoIndex],
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          if (NewOpSigned)
            return IRB.CreateICmpSLT(LHS, RHS);
          return IRB.CreateICmpULT(LHS, RHS);
        }).first;
      break;
    case DirectiveLoopInfo::LE:
      LoopCmp = buildInstructionSignDependent(
        IRB, M, IndVarVal, LoopInfo.UBound[LInfoIndex].Result, LoopInfo.IndVarSigned[LInfoIndex], LoopInfo.UBoundSigned[LInfoIndex],
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          if (NewOpSigned)
            return IRB.CreateICmpSLE(LHS, RHS);
          return IRB.CreateICmpULE(LHS, RHS);
        }).first;
      break;
    case DirectiveLoopInfo::GT:
      LoopCmp = buildInstructionSignDependent(
        IRB, M, IndVarVal, LoopInfo.UBound[LInfoIndex].Result, LoopInfo.IndVarSigned[LInfoIndex], LoopInfo.UBoundSigned[LInfoIndex],
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          if (NewOpSigned)
            return IRB.CreateICmpSGT(LHS, RHS);
          return IRB.CreateICmpUGT(LHS, RHS);
        }).first;
      break;
    case DirectiveLoopInfo::GE:
      LoopCmp = buildInstructionSignDependent(
        IRB, M, IndVarVal, LoopInfo.UBound[LInfoIndex].Result, LoopInfo.IndVarSigned[LInfoIndex], LoopInfo.UBoundSigned[LInfoIndex],
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          if (NewOpSigned)
            return IRB.CreateICmpSGE(LHS, RHS);
          return IRB.CreateICmpUGE(LHS, RHS);
        }).first;
      break;
    default:
      llvm_unreachable("unexpected loop type");
    }

    BasicBlock *BodyBB = Entry->getParent()->splitBasicBlock(Entry);
    BasicBlock *CollapseBB = BodyBB;
    for (size_t i = 0; i < LoopInfo.LBound.size(); ++i) {
      CollapseBB = CollapseBB->splitBasicBlock(Entry);
      CollapseStuff.push_back(&CollapseBB->getUniquePredecessor()->front());
    }
    CollapseBB->setName("for.body");

    BasicBlock *EndBB = BasicBlock::Create(M.getContext(), "for.end", &F);
    IRB.SetInsertPoint(EndBB);
    IRB.CreateBr(Exit->getParent()->getUniqueSuccessor());

    // The new exit is the end of the loop
    LoopExitI = &EndBB->front();

    // Replace default br. by a conditional br. to task.body or task end
    Instruction *OldTerminator = CondBB->getTerminator();
    IRB.SetInsertPoint(OldTerminator);
    IRB.CreateCondBr(LoopCmp, BodyBB, EndBB);
    OldTerminator->eraseFromParent();

    BasicBlock *IncrBB = BasicBlock::Create(M.getContext(), "for.incr", &F);

    // Add a br. to for.cond
    IRB.SetInsertPoint(IncrBB);
    IndVarVal = IRB.CreateLoad(
        IndVarTy,
        LoopInfo.IndVar[LInfoIndex]);
    auto p = buildInstructionSignDependent(
      IRB, M, IndVarVal, LoopInfo.Step[LInfoIndex].Result, LoopInfo.IndVarSigned[LInfoIndex], LoopInfo.StepSigned[LInfoIndex],
      [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
        return IRB.CreateAdd(LHS, RHS);
      });
    IRB.CreateStore(createZSExtOrTrunc(IRB, p.first, IndVarTy, p.second), LoopInfo.IndVar[LInfoIndex]);
    IRB.CreateBr(CondBB);

    // Replace task end br. by a br. to for.incr
    OldTerminator = Exit->getParent()->getTerminator();
    assert(OldTerminator->getNumSuccessors() == 1);
    OldTerminator->setSuccessor(0, IncrBB);
  }

  Instruction *buildLoopForTask(
      Module &M, Function &F, const DirectiveEnvironment &DirEnv, Instruction *Entry,
      Instruction *Exit, const DirectiveLoopInfo &LoopInfo) {
    Instruction *LoopEntryI = nullptr;
    Instruction *LoopExitI = nullptr;

    SmallVector<Instruction *> CollapseStuff;
    for (int i = LoopInfo.LBound.size() - 1; i >= 0; --i) {
      IRBuilder<> IRB(Entry);
      Type *IndVarTy = DirEnv.getDSAType(DirEnv.LoopInfo.IndVar[i]);
      LoopInfo.LBound[i].Result = IRB.CreateCall(LoopInfo.LBound[i].Fun, LoopInfo.LBound[i].Args);
      LoopInfo.UBound[i].Result = IRB.CreateCall(LoopInfo.UBound[i].Fun, LoopInfo.UBound[i].Args);
      LoopInfo.Step[i].Result = IRB.CreateCall(LoopInfo.Step[i].Fun, LoopInfo.Step[i].Args);

      buildLoopForTaskImpl(M, F, IndVarTy, Entry, Exit, LoopInfo, LoopEntryI, LoopExitI, CollapseStuff, i);
      Entry = LoopEntryI;
      Exit = LoopExitI;
    }
    return LoopEntryI;
  }

  // Build a loop containing Single Entry Single Exit region Entry/Exit
  // and returns the WhileEntry and WhileExit
  void buildWhileForTaskImpl(Module &M, Function &F, Instruction *Entry,
                        Instruction *Exit, const DirectiveWhileInfo &WhileInfo,
                         Instruction *&WhileEntryI, Instruction *&WhileExitI) {
    BasicBlock *CondBB = Entry->getParent()->splitBasicBlock(Entry);
    CondBB->setName("while.cond");

    // The new entry is the start of the loop
    WhileEntryI = &CondBB->getUniquePredecessor()->front();

    IRBuilder<> IRB(Entry);

    WhileInfo.Result = IRB.CreateCall(WhileInfo.Fun, WhileInfo.Args);

    BasicBlock *BodyBB = Entry->getParent()->splitBasicBlock(Entry);
    BodyBB->setName("while.body");

    BasicBlock *EndBB = BasicBlock::Create(M.getContext(), "while.end", &F);
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
      Module &M, Function &F, Instruction *Entry,
      Instruction *Exit, const DirectiveWhileInfo &WhileInfo) {
    Instruction *WhileEntryI = nullptr;
    Instruction *WhileExitI = nullptr;

    buildWhileForTaskImpl(M, F, Entry, Exit, WhileInfo, WhileEntryI, WhileExitI);
    return WhileEntryI;
  }

  void buildLoopForMultiDep(
      Module &M, Function &F, const DirectiveEnvironment &DirEnv, Instruction *Entry, Instruction *Exit,
      const MultiDependInfo *MultiDepInfo) {
    DenseMap<Value *, Value *> IndToRemap;
    for (size_t i = 0; i < MultiDepInfo->Iters.size(); i++) {
      Value *IndVar = MultiDepInfo->Iters[i];
      buildLoopForMultiDepImpl(
        M, F, DirEnv, Entry, Exit, IndVar, i, MultiDepInfo->ComputeMultiDepFun,
        MultiDepInfo->Args, IndToRemap);
    }
  }

  void buildLoopForMultiDepImpl(
      Module &M, Function &F, const DirectiveEnvironment &DirEnv, Instruction *Entry, Instruction *Exit,
      Value *IndVar, int IterSelector, Function *ComputeMultiDepFun,
      ArrayRef<Value *> Args, DenseMap<Value *, Value *> &IndToRemap) {

    DenseSet<Instruction *> InstrToRemap;

    SmallVector<Value *, 4> TmpArgs(Args.begin(), Args.end());
    // Select what iterator info we want to be computed
    TmpArgs.push_back(ConstantInt::get(Type::getInt64Ty(M.getContext()), IterSelector));

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

    BasicBlock *IncrBB = BasicBlock::Create(M.getContext(), "for.incr", &F);

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
  void registerTaskInfo(Module &M, Value *TaskInfoVar) {
    Function *Func = cast<Function>(TaskInfoRegisterCtorFuncCallee.getCallee());
    BasicBlock &Entry = Func->getEntryBlock();

    IRBuilder<> BBBuilder(&Entry.getInstList().back());
    BBBuilder.CreateCall(TaskInfoRegisterFuncCallee, TaskInfoVar);
  }

  void registerAssert(Module &M, StringRef Str) {
    Function *Func = cast<Function>(RegisterCtorAssertFuncCallee.getCallee());
    BasicBlock &Entry = Func->getEntryBlock();

    IRBuilder<> BBBuilder(&Entry.getInstList().back());
    Constant *StringPtr = BBBuilder.CreateGlobalStringPtr(Str);
    BBBuilder.CreateCall(RegisterAssertFuncCallee, StringPtr);
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
      Type *Ty = DirInfo.DirEnv.getDSAType(V);
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
      Function *F, const DirectiveLoopInfo &LoopInfo,
      ArrayRef<Value *> NewIndVarLBound, ArrayRef<Value *> NewIndVarUBound, bool IsTaskLoop,
      CallInst *& CallComputeDepStart, CallInst *& CallComputeDepEnd,
      CallInst *& RegisterDepCall) {

    BasicBlock &Entry = F->getEntryBlock();
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
      TaskDepAPICall.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()), DRI.lookup(Base).ReductionIndex));
    }

    Constant *RegionTextGV = BBBuilder.CreateGlobalStringPtr(DepInfo->RegionText);

    Value *Handler = &*(F->arg_end() - 1);
    TaskDepAPICall.push_back(Handler);
    TaskDepAPICall.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()), DepInfo->SymbolIndex));
    TaskDepAPICall.push_back(BBBuilder.CreateBitCast(RegionTextGV, Type::getInt8PtrTy(M.getContext())));
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
      Function *F, const DirectiveLoopInfo &LoopInfo, ArrayRef<Value *> NewIndVarLBound, ArrayRef<Value *> NewIndVarUBound,
      bool IsTaskLoop) {

    CallInst *CallComputeDepStart;
    CallInst *CallComputeDepEnd;
    CallInst *RegisterDepCall;
    unpackDepCallToRTImpl(
      M, DepInfo, DRI, F, LoopInfo, NewIndVarLBound, NewIndVarUBound, IsTaskLoop,
      CallComputeDepStart, CallComputeDepEnd, RegisterDepCall);
  }

  void unpackMultiRangeCall(
      Module &M, const DirectiveEnvironment &DirEnv,
      const MultiDependInfo *MultiDepInfo,
      const DirectiveReductionsInitCombInfo &DRI,
      Function *F, const DirectiveLoopInfo &LoopInfo, ArrayRef<Value *> NewIndVarLBound, ArrayRef<Value *> NewIndVarUBound,
      bool IsTaskLoop) {

    CallInst *CallComputeDepStart;
    CallInst *CallComputeDepEnd;
    CallInst *RegisterDepCall;
    unpackDepCallToRTImpl(
      M, MultiDepInfo, DRI, F, LoopInfo, NewIndVarLBound, NewIndVarUBound, IsTaskLoop,
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
        M, *F, DirEnv, CallComputeDepStart, RegisterDepCall, IndVar, i, MultiDepInfo->ComputeMultiDepFun,
        Args, IndToRemap);
    }
  }

  void unpackDepsCallToRT(
      Module &M, const DirectiveInfo &DirInfo, Function *F,
      bool IsTaskLoop, ArrayRef<Value *> NewIndVarLBound, ArrayRef<Value *> NewIndVarUBound) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveLoopInfo &LoopInfo = DirEnv.LoopInfo;
    const DirectiveReductionsInitCombInfo &DRI = DirEnv.ReductionsInitCombInfo;
    const DirectiveDependsInfo &DependsInfo = DirEnv.DependsInfo;

    for (auto &DepInfo : DependsInfo.List) {
      if (auto *MultiDepInfo = dyn_cast<MultiDependInfo>(DepInfo.get())) {
        // Multideps using loop iterator are assumed to be discrete
        if (IsTaskLoop && multidepUsesLoopIter(LoopInfo, *MultiDepInfo)) {
          unpackMultiRangeCall(
            M, DirInfo.DirEnv, MultiDepInfo, DRI, F,
            LoopInfo, NewIndVarLBound, NewIndVarLBound, IsTaskLoop);
        } else {
          unpackMultiRangeCall(
            M, DirInfo.DirEnv, MultiDepInfo, DRI, F,
            LoopInfo, NewIndVarLBound, NewIndVarUBound, IsTaskLoop);
        }
      } else {
        unpackDepCallToRT(
          M, DepInfo.get(), DRI, F,
          LoopInfo, NewIndVarLBound, NewIndVarUBound, IsTaskLoop);
      }
    }
  }

  void unpackDepsAndRewrite(Module &M, const DirectiveInfo &DirInfo,
                            Function *F,
                            const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock &Entry = F->getEntryBlock();

    // add the terminator so IRBuilder inserts just before it
    F->getEntryBlock().getInstList().push_back(ReturnInst::Create(M.getContext()));

    SmallVector<Value *, 2> NewIndVarLBounds;
    SmallVector<Value *, 2> NewIndVarUBounds;

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveLoopInfo &LoopInfo = DirEnv.LoopInfo;

    bool IsTaskLoop = DirEnv.isOmpSsTaskLoopDirective();

    if (IsTaskLoop) {
      Type *IndVarTy = DirInfo.DirEnv.getDSAType(LoopInfo.IndVar[0]);

      IRBuilder<> IRB(&Entry.front());
      Value *LoopBounds = &*(F->arg_end() - 2);

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
      Value *LBoundField = IRB.CreateGEP(
          Nanos6LoopBounds::getInstance(M).getType(),
          LoopBounds, Idx, "lb_gep");
      LBoundField = IRB.CreateLoad(
          Nanos6LoopBounds::getInstance(M).getType()->getElementType(1), LBoundField);
      LBoundField = IRB.CreateZExtOrTrunc(LBoundField, IndVarTy, "lb");

      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 1);
      Value *UBoundField = IRB.CreateGEP(
          Nanos6LoopBounds::getInstance(M).getType(),
          LoopBounds, Idx, "ub_gep");
      UBoundField = IRB.CreateLoad(
          Nanos6LoopBounds::getInstance(M).getType()->getElementType(2), UBoundField);
      UBoundField = IRB.CreateZExtOrTrunc(UBoundField, IndVarTy);
      UBoundField = IRB.CreateSub(UBoundField, ConstantInt::get(IndVarTy, 1), "ub");

      SmallVector<Value *> NormalizedUBs(LoopInfo.UBound.size());
      // This is used for nanos6_create_loop
      // NOTE: all values have nanos6 upper_bound type
      ComputeLoopBounds(M, LoopInfo, &Entry.back(), NormalizedUBs);

      for (size_t i = 0; i < LoopInfo.LBound.size(); ++i) {
        Value *NewIndVarLBound = IRB.CreateAlloca(IndVarTy, nullptr, LoopInfo.IndVar[i]->getName() + ".lb");
        Value *NewIndVarUBound = IRB.CreateAlloca(IndVarTy, nullptr, LoopInfo.IndVar[i]->getName() + ".ub");

        auto f = [](Module &M, IRBuilder<> &IRB, ArrayRef<Value *> NormalizedUBs, const DirectiveLoopInfo &LoopInfo, int i, Value *Val, Type *Ty, Value *Storage) {
          // NOTE: NormalizedUBs values are nanos6_register_loop upper_bound type
          Value *Niters = ConstantInt::get(Nanos6LoopBounds::getInstance(M).getType()->getElementType(1), 1);
          for (size_t j = i + 1; j < LoopInfo.LBound.size(); ++j)
            Niters = IRB.CreateMul(Niters, NormalizedUBs[j]);

          // TMP = (LB|UB)/ProductUBs(i+1..n)
          auto pTmp = buildInstructionSignDependent(
            IRB, M, Val, Niters, Ty, /*RHSSigned=*/false,
            [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
              if (NewOpSigned)
                return IRB.CreateSDiv(LHS, RHS);
              return IRB.CreateUDiv(LHS, RHS);
            });

          // TMP * Step
          auto p = buildInstructionSignDependent(
            IRB, M, pTmp.first, LoopInfo.Step[i].Result, pTmp.second, LoopInfo.StepSigned[i],
            [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
              return IRB.CreateMul(LHS, RHS);
            });

          // (TMP * Step) + OrigLBound
          auto pBodyIndVar = buildInstructionSignDependent(
            IRB, M, p.first, LoopInfo.LBound[i].Result, p.second, LoopInfo.LBoundSigned[i],
            [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
              return IRB.CreateAdd(LHS, RHS);
            });

          // TMP*ProductUBs(i+1..n)
          p = buildInstructionSignDependent(
            IRB, M, pTmp.first, Niters, pTmp.second, /*RHSSigned=*/false,
            [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
              return IRB.CreateMul(LHS, RHS);
            });

          // LoopInfo - TMP*ProductUBs(i+1..n)
          auto pLoopIndVar = buildInstructionSignDependent(
            IRB, M, Val, p.first, Ty, p.second,
            [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
              return IRB.CreateSub(LHS, RHS);
            });

          // (LB|UB) = LoopInfo - TMP*ProductUBs(i+1..n)
          Val = pLoopIndVar.first;
          // BodyIndVar(i) = (TMP * Step) + OrigLBound
          IRB.CreateStore(createZSExtOrTrunc(IRB, pBodyIndVar.first, Ty, pBodyIndVar.second), Storage);
          return Val;
        };

        LBoundField = f(M, IRB, NormalizedUBs, LoopInfo, i, LBoundField, IndVarTy, NewIndVarLBound);
        UBoundField = f(M, IRB, NormalizedUBs, LoopInfo, i, UBoundField, IndVarTy, NewIndVarUBound);

        NewIndVarLBounds.push_back(NewIndVarLBound);
        NewIndVarUBounds.push_back(NewIndVarUBound);
      }
    }

    // Insert RT call before replacing uses
    unpackDepsCallToRT(M, DirInfo, F, IsTaskLoop, NewIndVarLBounds, NewIndVarUBounds);

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

  void unpackCostAndRewrite(Module &M, const DirectiveCostInfo &CostInfo, Function *F,
                            const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock &Entry = F->getEntryBlock();
    F->getEntryBlock().getInstList().push_back(ReturnInst::Create(M.getContext()));
    IRBuilder<> BBBuilder(&F->getEntryBlock().back());
    Value *Constraints = &*(F->arg_end() - 1);
    Value *Idx[2];
    Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
    Idx[1] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));

    Value *GEPConstraints =
        BBBuilder.CreateGEP(Nanos6TaskConstraints::getInstance(M).getType(),
                            Constraints, Idx, "gep_" + Constraints->getName());
    Value *Cost = BBBuilder.CreateCall(CostInfo.Fun, CostInfo.Args);
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

  void unpackPriorityAndRewrite(
      Module &M, const DirectivePriorityInfo &PriorityInfo, Function *F,
      const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock &Entry = F->getEntryBlock();
    F->getEntryBlock().getInstList().push_back(ReturnInst::Create(M.getContext()));
    IRBuilder<> BBBuilder(&F->getEntryBlock().back());
    Value *PriorityArg = &*(F->arg_end() - 1);

    Value *Priority = BBBuilder.CreateCall(PriorityInfo.Fun, PriorityInfo.Args);
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

  void unpackOnreadyAndRewrite(
      Module &M, const DirectiveOnreadyInfo &OnreadyInfo, Function *F,
      const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock &Entry = F->getEntryBlock();
    F->getEntryBlock().getInstList().push_back(ReturnInst::Create(M.getContext()));
    IRBuilder<> BBBuilder(&F->getEntryBlock().back());

    BBBuilder.CreateCall(OnreadyInfo.Fun, OnreadyInfo.Args);
    for (Instruction &I : Entry) {
      Function::arg_iterator AI = F->arg_begin();
      for (auto It = StructToIdxMap.begin();
             It != StructToIdxMap.end(); ++It, ++AI) {
        if (isReplaceableValue(It->first))
          I.replaceUsesOfWith(It->first, &*AI);
      }
    }
  }

  void unpackWhileAndRewrite(
      Module &M, const DirectiveWhileInfo &WhileInfo, Function *F,
      const MapVector<Value *, size_t> &StructToIdxMap) {
    BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock &Entry = F->getEntryBlock();
    F->getEntryBlock().getInstList().push_back(ReturnInst::Create(M.getContext()));
    IRBuilder<> BBBuilder(&F->getEntryBlock().back());
    Value *ResArg = &*(F->arg_end() - 1);

    Value *Res = BBBuilder.CreateCall(WhileInfo.Fun, WhileInfo.Args);
    Value *ResSExt = BBBuilder.CreateZExt(Res, Type::getInt8Ty(M.getContext()));
    BBBuilder.CreateStore(ResSExt, ResArg);

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
                                 ArrayRef<StringRef> ExtraNameList,
                                 bool IsTask = false) {
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

    // Build debug info in task unpack
    if (IsTask) {
      DISubprogram *OldSP = F.getSubprogram();
      DIBuilder DIB(M, /*AllowUnresolved=*/false, OldSP->getUnit());
      auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray(None));
      DISubprogram::DISPFlags SPFlags = DISubprogram::SPFlagDefinition |
                                        DISubprogram::SPFlagOptimized |
                                        DISubprogram::SPFlagLocalToUnit;
      DISubprogram *NewSP = DIB.createFunction(
          OldSP->getUnit(), FuncVar->getName(), FuncVar->getName(), OldSP->getFile(),
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
      IRBuilder<> &IRBTranslate, IRBuilder<> &IRBReload, const DependInfo *DepInfo, Value *DSA,
      bool IsShared,
      Value *&UnpackedDSA, Value *AddrTranslationTable, Type *AddrTranslationTableTy, int SymbolIndex) {

    llvm::Value *DepBase = UnpackedDSA;
    if (!IsShared) {
      assert(!isa<LoadInst>(UnpackedDSA));
      DepBase = IRBTranslate.CreateLoad(
        IRBTranslate.getPtrTy(), DepBase);
    }

    Value *Idx[2];
    Idx[0] = ConstantInt::get(Type::getInt32Ty(
      IRBTranslate.getContext()), SymbolIndex);
    Idx[1] = Constant::getNullValue(Type::getInt32Ty(IRBTranslate.getContext()));
    Value *LocalAddr = IRBTranslate.CreateGEP(
        AddrTranslationTableTy,
        AddrTranslationTable, Idx, "local_lookup_" + DSA->getName());
    LocalAddr = IRBTranslate.CreateLoad(
        IRBTranslate.getInt64Ty(), LocalAddr);

    Idx[1] = ConstantInt::get(Type::getInt32Ty(IRBTranslate.getContext()), 1);
    Value *DeviceAddr = IRBTranslate.CreateGEP(
        AddrTranslationTableTy,
        AddrTranslationTable, Idx, "device_lookup_" + DSA->getName());
    DeviceAddr = IRBTranslate.CreateLoad(
        IRBTranslate.getInt64Ty(), DeviceAddr);

    // Res = device_addr + (DSA_addr - local_addr)
    Value *Translation =
        IRBTranslate.CreateGEP(IRBTranslate.getInt8Ty(),
                               DepBase, IRBTranslate.CreateNeg(LocalAddr));
    Translation =
        IRBTranslate.CreateGEP(IRBTranslate.getInt8Ty(),
                               Translation, DeviceAddr);

    // Store the translation
    if (IsShared) {
      auto *LUnpackedDSA = cast<LoadInst>(UnpackedDSA);
      Translation = IRBTranslate.CreateBitCast(Translation, LUnpackedDSA->getType());
      IRBTranslate.CreateStore(Translation, LUnpackedDSA->getPointerOperand());
      // Reload what we have translated
      UnpackedDSA = IRBReload.CreateLoad(
          IRBReload.getPtrTy(),
          LUnpackedDSA->getPointerOperand());
    } else {
      IRBTranslate.CreateStore(Translation, UnpackedDSA);
    }
  }

  // Given a Outline Function assuming that task args are the first parameter, and
  // DSAInfo and VLADimsInfo, it unpacks task args in Outline and fills UnpackedList
  // with those Values, used to call Unpack Functions
  void unpackDSAsWithVLADims(Module &M, const DirectiveInfo &DirInfo,
                  Type *TaskArgsTy,
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
          TaskArgsTy,
          OlDepsFuncTaskArgs, Idx, "gep_" + V->getName());
      Value *LGEP =
          BBBuilder.CreateLoad(PointerType::getUnqual(M.getContext()), GEP,
                               "load_" + GEP->getName());

      UnpackedList.push_back(LGEP);
    }
    for (Value *V : DSAInfo.Private) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), StructToIdxMap.lookup(V));
      Value *GEP = BBBuilder.CreateGEP(
          TaskArgsTy,
          OlDepsFuncTaskArgs, Idx, "gep_" + V->getName());

      // VLAs
      if (VLADimsInfo.count(V))
        GEP = BBBuilder.CreateLoad(PointerType::getUnqual(M.getContext()), GEP, "load_" + GEP->getName());

      UnpackedList.push_back(GEP);
    }
    for (Value *V : DSAInfo.Firstprivate) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), StructToIdxMap.lookup(V));
      Value *GEP = BBBuilder.CreateGEP(
          TaskArgsTy,
          OlDepsFuncTaskArgs, Idx, "gep_" + V->getName());

      // VLAs
      if (VLADimsInfo.count(V))
        GEP = BBBuilder.CreateLoad(PointerType::getUnqual(M.getContext()), GEP, "load_" + GEP->getName());

      UnpackedList.push_back(GEP);
    }
    for (Value *V : CapturedInfo) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), StructToIdxMap.lookup(V));
      Value *GEP = BBBuilder.CreateGEP(
          TaskArgsTy,
          OlDepsFuncTaskArgs, Idx, "capt_gep" + V->getName());
      Value *LGEP =
          BBBuilder.CreateLoad(V->getType(), GEP, "load_" + GEP->getName());
      UnpackedList.push_back(LGEP);
    }
  }

  // task_args cannot be modified. This function creates
  // new variables where the translation is performed.
  void dupTranlationNeededArgs(
      IRBuilder<> &IRBEntry,
      const std::map<Value *, std::pair<const DependInfo *, int>> &DepSymToIdx,
      const MapVector<Value *, size_t> &StructToIdxMap,
      SmallVector<Value *, 4> &UnpackParams,
      bool HasDevice) {

    for (const auto &p : DepSymToIdx) {
      Value *DepBaseDSA = p.first;
      const DependInfo *DI = p.second.first;
      size_t Idx = StructToIdxMap.lookup(DepBaseDSA);
      if (HasDevice)
        Idx -= DeviceArgsSize;
      Value *UnpackedDSA = UnpackParams[Idx];
      if (auto *LUnpackedDSA = dyn_cast<LoadInst>(UnpackedDSA)) {
        Value *NewDepBaseDSA =
          IRBEntry.CreateAlloca(
            IRBEntry.getPtrTy(), nullptr, "tlate." + LUnpackedDSA->getName());
        IRBEntry.CreateStore(LUnpackedDSA, NewDepBaseDSA);

        UnpackParams[Idx] =
          IRBEntry.CreateLoad(IRBEntry.getPtrTy(), NewDepBaseDSA);
      } else {
        Value *NewDepBaseDSA =
          IRBEntry.CreateAlloca(
            IRBEntry.getPtrTy(), nullptr, "tlate." + UnpackedDSA->getName());
        IRBEntry.CreateStore(
          IRBEntry.CreateLoad(
            IRBEntry.getPtrTy(), UnpackedDSA),
          NewDepBaseDSA);

        UnpackParams[Idx] = NewDepBaseDSA;
      }
    }
  }

  // Given an Outline and Unpack Functions it unpacks DSAs in Outline
  // and builds a call to Unpack
  void olCallToUnpack(Module &M, const DirectiveInfo &DirInfo,
                      Type *TaskArgsTy,
                      const MapVector<Value *, size_t> &StructToIdxMap,
                      Function *OlFunc, Function *UnpackFunc,
                      Type *AddrTranslationTableTy,
                      bool IsTaskFunc=false) {
    BasicBlock::Create(M.getContext(), "entry", OlFunc);

    // First arg is the nanos_task_args
    Function::arg_iterator AI = OlFunc->arg_begin();
    AI++;
    SmallVector<Value *, 4> UnpackParams;
    unpackDSAsWithVLADims(M, DirInfo, TaskArgsTy, OlFunc, StructToIdxMap, UnpackParams);
    while (AI != OlFunc->arg_end()) {
      UnpackParams.push_back(&*AI++);
    }

    if (IsTaskFunc) {
      // Build call to compute_dep in order to have get the base dependency of
      // the reduction. The result is passed to unpack
      const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
      // Preserve the params before translation. And replace used after build all
      // compute_dep calls
      // NOTE: this assumes UnpackParams can be indexed with StructToIdxMap
      // IRBuilder<> BBBuilder(&OlFunc->getEntryBlock());

      BasicBlock *IfThenBB = BasicBlock::Create(M.getContext(), "", OlFunc);
      BasicBlock *IfEndBB = BasicBlock::Create(M.getContext(), "", OlFunc);

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

      bool HasDevice = !DirInfo.DirEnv.DeviceInfo.empty();

      dupTranlationNeededArgs(
          IRBEntry, DirInfo.DirEnv.DepSymToIdx, StructToIdxMap, UnpackParams, HasDevice);

      SmallVector<Value *, 4> UnpackParamsCopy(UnpackParams);

      for (const auto &p : DirInfo.DirEnv.DepSymToIdx) {
        Value *DepBaseDSA = p.first;
        const DependInfo *DI = p.second.first;
        int SymbolIndex = p.second.second;

        bool IsShared = DirInfo.DirEnv.DSAInfo.Shared.count(DepBaseDSA);

        size_t Idx = StructToIdxMap.lookup(DepBaseDSA);
        if (HasDevice)
          Idx -= DeviceArgsSize;

        translateDep(
          IRBIfThen, IRBIfEnd, DI, DepBaseDSA,
          IsShared,
          UnpackParams[Idx],
          AddrTranslationTable, AddrTranslationTableTy, SymbolIndex);
      }
      // Replaces dsa uses by unpacked values
      for (Instruction &I : *IfThenBB) {
        auto UnpackedIt = UnpackParamsCopy.begin();
        for (auto It = StructToIdxMap.begin();
               It != StructToIdxMap.end(); ++It, ++UnpackedIt) {
          if (isReplaceableValue(It->first))
            I.replaceUsesOfWith(It->first, *UnpackedIt);
        }
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
    Value *TaskArgsDstL = IRB.CreateLoad(PointerType::getUnqual(M.getContext()), TaskArgsDst);

    SmallVector<VLAAlign, 2> VLAAlignsInfo;
    computeVLAsAlignOrder(M, VLAAlignsInfo, DirEnv, VLADimsInfo);

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

    Value *TaskArgsDstLi8IdxGEP =
      IRB.CreateGEP(IRB.getInt8Ty(), TaskArgsDstL, TaskArgsStructSizeOf, "args_end");

    // First point VLAs to its according space in task args
    for (const VLAAlign& VAlign : VLAAlignsInfo) {
      auto *V = VAlign.V;
      Type *Ty = DirEnv.getDSAType(V);
      unsigned TyAlign = VAlign.Align;

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
      Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(V));
      Value *GEP =
          IRB.CreateGEP(TaskArgsTy,
                        TaskArgsDstL, Idx, "gep_dst_" + V->getName());

      // Point VLA in task args to an aligned position of the extra space allocated
      IRB.CreateAlignedStore(TaskArgsDstLi8IdxGEP, GEP, Align(TyAlign));
      // Skip current VLA size
      unsigned SizeB = M.getDataLayout().getTypeAllocSize(Ty);
      Value *VLASize = ConstantInt::get(IRB.getInt64Ty(), SizeB);
      for (auto *Dim : VLADimsInfo.lookup(V)) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(Dim));
        Value *GEPDst =
            IRB.CreateGEP(TaskArgsTy,
                          TaskArgsDstL, Idx, "gep_dst_" + Dim->getName());
        GEPDst =
            IRB.CreateLoad(IRB.getInt64Ty(), GEPDst);
        VLASize = IRB.CreateNUWMul(VLASize, GEPDst);
      }
      TaskArgsDstLi8IdxGEP = IRB.CreateGEP(
          IRB.getInt8Ty(),
          TaskArgsDstLi8IdxGEP, VLASize);
    }

    for (size_t i = 0; i < DSAInfo.Shared.size(); ++i) {
      Value *V = DSAInfo.Shared[i];

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
      Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(V));
      Value *GEPSrc = IRB.CreateGEP(
          TaskArgsTy,
          TaskArgsSrc, Idx, "gep_src_" + V->getName());
      Value *GEPDst = IRB.CreateGEP(
          TaskArgsTy,
          TaskArgsDstL, Idx, "gep_dst_" + V->getName());
      IRB.CreateStore(
          IRB.CreateLoad(PointerType::getUnqual(M.getContext()), GEPSrc),
          GEPDst);
    }
    for (size_t i = 0; i < DSAInfo.Private.size(); ++i) {
      Value *V = DSAInfo.Private[i];
      Type *Ty = DSAInfo.PrivateTy[i];
      // Call custom constructor generated in clang in non-pods
      // Leave pods unititialized
      auto It = NonPODsInfo.Inits.find(V);
      if (It != NonPODsInfo.Inits.end()) {
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
            Value *GEPSrc =
                IRB.CreateGEP(TaskArgsTy,
                              TaskArgsSrc, Idx, "gep_src_" + Dim->getName());
            GEPSrc = IRB.CreateLoad(IRB.getInt64Ty(), GEPSrc);
            NSize = IRB.CreateNUWMul(NSize, GEPSrc);
          }
        }

        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(V));
        Value *GEP =
            IRB.CreateGEP(TaskArgsTy,
                          TaskArgsDstL, Idx, "gep_" + V->getName());

        // VLAs
        if (VLADimsInfo.count(V))
          GEP = IRB.CreateLoad(PointerType::getUnqual(M.getContext()), GEP);

        IRB.CreateCall(FunctionCallee(cast<Function>(It->second)), ArrayRef<Value*>{GEP, NSize});
      }
    }
    for (size_t i = 0; i < DSAInfo.Firstprivate.size(); ++i) {
      Value *V = DSAInfo.Firstprivate[i];
      Type *Ty = DSAInfo.FirstprivateTy[i];
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
          Value *GEPSrc =
              IRB.CreateGEP(TaskArgsTy,
                            TaskArgsSrc, Idx, "gep_src_" + Dim->getName());
          // VLA sizes are always i64
          GEPSrc = IRB.CreateLoad(IRB.getInt64Ty(), GEPSrc);
          NSize = IRB.CreateNUWMul(NSize, GEPSrc);
        }
      }

      // call custom copy constructor generated in clang in non-pods
      // do a memcpy if pod
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
      Idx[1] = ConstantInt::get(IRB.getInt32Ty(), StructToIdxMap.lookup(V));
      Value *GEPSrc =
          IRB.CreateGEP(TaskArgsTy,
                        TaskArgsSrc, Idx, "gep_src_" + V->getName());
      Value *GEPDst =
          IRB.CreateGEP(TaskArgsTy,
                        TaskArgsDstL, Idx, "gep_dst_" + V->getName());

      // VLAs
      if (VLADimsInfo.count(V)) {
        GEPSrc =
            IRB.CreateLoad(PointerType::getUnqual(M.getContext()), GEPSrc);
        GEPDst =
            IRB.CreateLoad(PointerType::getUnqual(M.getContext()), GEPDst);
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

  Value *computeTaskArgsVLAsExtraSizeOf(
      Module &M, const DirectiveEnvironment &DirEnv, IRBuilder<> &IRB, const DirectiveVLADimsInfo &VLADimsInfo) {
    Value *Sum = ConstantInt::get(IRB.getInt64Ty(), 0);
    for (const auto &VLAWithDimsMap : VLADimsInfo) {
      Type *Ty = DirEnv.getDSAType(VLAWithDimsMap.first);
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
    const DirectiveDeviceInfo &DeviceInfo = DirEnv.DeviceInfo;
    SmallVector<Type *, 4> TaskArgsMemberTy;
    size_t TaskArgsIdx = 0;

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
        TaskArgsMemberTy.push_back(Type::getInt64Ty(M.getContext()));
        TaskArgsIdx++;
      }
      // size_t shm_size;
      TaskArgsMemberTy.push_back(Type::getInt64Ty(M.getContext()));
      TaskArgsIdx++;
    }

    // Private and Firstprivate must be stored in the struct
    // Captured values (i.e. VLA dimensions) are not pointers
    for (Value *V : DSAInfo.Shared) {
      TaskArgsMemberTy.push_back(PointerType::getUnqual(M.getContext()));
      StructToIdxMap[V] = TaskArgsIdx++;
    }
    for (size_t i = 0; i < DSAInfo.Private.size(); ++i) {
      Value *V = DSAInfo.Private[i];
      Type *Ty = DSAInfo.PrivateTy[i];
      // VLAs
      if (VLADimsInfo.count(V))
        TaskArgsMemberTy.push_back(PointerType::getUnqual(M.getContext()));
      else
        TaskArgsMemberTy.push_back(Ty);
      StructToIdxMap[V] = TaskArgsIdx++;
    }
    for (size_t i = 0; i < DSAInfo.Firstprivate.size(); ++i) {
      Value *V = DSAInfo.Firstprivate[i];
      Type *Ty = DSAInfo.FirstprivateTy[i];
      // VLAs
      if (VLADimsInfo.count(V))
        TaskArgsMemberTy.push_back(PointerType::getUnqual(M.getContext()));
      else
        TaskArgsMemberTy.push_back(Ty);
      StructToIdxMap[V] = TaskArgsIdx++;
    }
    for (Value *V : CapturedInfo) {
      assert(!V->getType()->isPointerTy() && "Captures are not pointers");
      TaskArgsMemberTy.push_back(V->getType());
      StructToIdxMap[V] = TaskArgsIdx++;
    }
    return StructType::create(M.getContext(), TaskArgsMemberTy, Str);
  }

  // Useful to get lists of info needed for creating functions, assign
  // names to the values and so on.
  // Also computes the list of sizeofs and offsets for the members
  // needed for devices.
  void getTaskArgsInfo(
      const DirectiveEnvironment &DirEnv,
      Module &M, Function &F,
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      size_t taskNum, StructType *TaskArgsTy,
      SmallVector<Type *, 4> &TaskTypeList, SmallVector<StringRef, 4> &TaskNameList,
      GlobalVariable *&SizeofTableVar, GlobalVariable *&OffsetTableVar,
      GlobalVariable *&ArgIdxTableVar) {

    const DirectiveDeviceInfo &DeviceInfo = DirEnv.DeviceInfo;

    for (auto It = TaskArgsToStructIdxMap.begin();
           It != TaskArgsToStructIdxMap.end(); ++It) {
      Value *V = It->first;
      TaskTypeList.push_back(V->getType());
      TaskNameList.push_back(V->getName());
    }

    // TODO: move this to a class member
    Type *Int32Ty = Type::getInt32Ty(M.getContext());

    // int *sizeof_table;
    // Type *SizeofTableTy = Nanos6TaskInfo::getInstance(M).getType()->getElementType(13);
    SizeofTableVar =
      cast<GlobalVariable>(M.getOrInsertGlobal(
        ("sizeof_table_var_" + F.getName() + Twine(taskNum)).str(),
        ArrayType::get(Int32Ty, TaskTypeList.size())));
    SizeofTableVar->setLinkage(GlobalVariable::InternalLinkage);
    SizeofTableVar->setAlignment(Align(64));
    SizeofTableVar->setConstant(true);
    SmallVector<Constant *, 4> TaskSizeofList;
    // TODO: Do this in a clever way. Remove the device elements
    ArrayRef<Type *> TaskElementTypes = TaskArgsTy->elements();
    if (!DeviceInfo.empty())
      TaskElementTypes = TaskElementTypes.drop_front(DeviceArgsSize);
    for (Type *Ty : TaskElementTypes) {
      TaskSizeofList.push_back(
        ConstantInt::get(
          Int32Ty,
          M.getDataLayout().getTypeStoreSize(Ty).getFixedSize()));
    }
    SizeofTableVar->setInitializer(
      ConstantArray::get(
        ArrayType::get(Int32Ty, TaskTypeList.size()),
        TaskSizeofList));

    // int *offset_table;
    // Type *OffsetTableTy = Nanos6TaskInfo::getInstance(M).getType()->getElementType(14);
    OffsetTableVar =
      cast<GlobalVariable>(M.getOrInsertGlobal(
        ("offset_table_var_" + F.getName() + Twine(taskNum)).str(),
        ArrayType::get(Int32Ty, TaskTypeList.size())));
    OffsetTableVar->setLinkage(GlobalVariable::InternalLinkage);
    OffsetTableVar->setAlignment(Align(64));
    OffsetTableVar->setConstant(true);
    ArrayRef<uint64_t> MemberOffsetsList =
      M.getDataLayout().getStructLayout(TaskArgsTy)->getMemberOffsets();
    if (!DeviceInfo.empty())
      MemberOffsetsList = MemberOffsetsList.drop_front(DeviceArgsSize);
    SmallVector<Constant *, 4> TaskOffsetList;
    for (const uint64_t &val : MemberOffsetsList) {
      TaskOffsetList.push_back(
        ConstantInt::get(Int32Ty, val));
    }
    OffsetTableVar->setInitializer(
      ConstantArray::get(
        ArrayType::get(Int32Ty, TaskTypeList.size()),
        TaskOffsetList));

    // int *arg_idx_table;
    // Type *ArgIdxTableTy = Nanos6TaskInfo::getInstance(M).getType()->getElementType(15);
    ArgIdxTableVar =
      cast<GlobalVariable>(M.getOrInsertGlobal(
        ("arg_idx_table_var_" + F.getName() + Twine(taskNum)).str(),
        ArrayType::get(Int32Ty, DirEnv.DependsInfo.NumSymbols)));
    ArgIdxTableVar->setLinkage(GlobalVariable::InternalLinkage);
    ArgIdxTableVar->setAlignment(Align(64));
    ArgIdxTableVar->setConstant(true);
    SmallVector<Constant *, 4> TaskArgIdxList(DirEnv.DependsInfo.NumSymbols);
    for (const auto &p : DirEnv.DepSymToIdx) {
      Value *DepBase = p.first;
      int SymbolIndex = p.second.second;
      int Idx = TaskArgsToStructIdxMap.lookup(DepBase);
      if (!DeviceInfo.empty())
        Idx -= DeviceArgsSize;
      TaskArgIdxList[SymbolIndex] =
        ConstantInt::get(Int32Ty, Idx);
    }
    ArgIdxTableVar->setInitializer(
      ConstantArray::get(
        ArrayType::get(Int32Ty, DirEnv.DependsInfo.NumSymbols),
        TaskArgIdxList));
  }

  struct VLAAlign {
    Value *V;
    unsigned Align;
  };

  // Greater alignemt go first
  void computeVLAsAlignOrder(
      Module &M, SmallVectorImpl<VLAAlign> &VLAAlignsInfo,
      const DirectiveEnvironment &DirEnv,
      const DirectiveVLADimsInfo &VLADimsInfo) {
    for (const auto &VLAWithDimsMap : VLADimsInfo) {
      auto *V = VLAWithDimsMap.first;
      Type *Ty = DirEnv.getDSAType(V);

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

  // Final codes need to float nested directives constant allocas to be placed
  // at the beginning of the final code
  // Loop directives need to float body constant allocas to the beginning of
  // the loop
  // This function does this assuming all the allocas are placed just
  // after the Entry intrinsic (because clang does that)
  void gatherConstAllocas(
      SmallVectorImpl<Instruction *> &Allocas, Instruction *Entry) {
    BasicBlock::iterator It = Entry->getParent()->begin();
    // Skip Entry intrinsic
    ++It;
    // Allocas inside the task are put just after Entry
    // intrinsic
    while (It != Entry->getParent()->end()) {
      if (auto *II = dyn_cast<AllocaInst>(&*It)) {
        if (isa<ConstantInt>(II->getArraySize()))
          Allocas.push_back(II);
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

  // This must be called before erasing original entry/exit
  void buildFinalCondCFG(
      Module &M, Function &F, const DirectiveInfo &DirInfo, ValueToValueMapTy &FinalInfo) {
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
      Instruction *OrigEntryI = InnerDirInfo.Entry;
      Instruction *OrigExitI = InnerDirInfo.Exit;
      Instruction *CloneEntryI = cast<Instruction>(FinalInfo.lookup(OrigEntryI));
      Instruction *CloneExitI = cast<Instruction>(FinalInfo.lookup(OrigExitI));

      gatherConstAllocas(Allocas, CloneEntryI);

      if (IsLoop) {
        DirectiveLoopInfo FinalLoopInfo = InnerDirInfo.DirEnv.LoopInfo;
        rewriteDirInfoForFinal(FinalLoopInfo, FinalInfo);
        buildLoopForTask(M, F, InnerDirInfo.DirEnv, CloneEntryI, CloneExitI, FinalLoopInfo);
      } else if (IsWhile) {
        DirectiveWhileInfo FinalWhileInfo = InnerDirInfo.DirEnv.WhileInfo;
        rewriteDirInfoForFinal(FinalWhileInfo, FinalInfo);
        buildWhileForTask(M, F, CloneEntryI, CloneExitI, FinalWhileInfo);
      }
    }

    Instruction *OrigEntryI = DirInfo.Entry;
    Instruction *OrigExitI = DirInfo.Exit;
    Instruction *CloneEntryI = cast<Instruction>(FinalInfo.lookup(OrigEntryI));
    Instruction *CloneExitI = cast<Instruction>(FinalInfo.lookup(OrigExitI));

    gatherConstAllocas(Allocas, CloneEntryI);

    Instruction *NewCloneEntryI = CloneEntryI;
    bool IsLoop =
      DirInfo.DirEnv.isOmpSsLoopDirective() ||
      DirInfo.DirEnv.isOmpSsTaskIterForDirective();
    bool IsWhile = DirInfo.DirEnv.isOmpSsTaskIterWhileDirective();
    if (IsLoop) {
      DirectiveLoopInfo FinalLoopInfo = DirInfo.DirEnv.LoopInfo;
      rewriteDirInfoForFinal(FinalLoopInfo, FinalInfo);
      NewCloneEntryI = buildLoopForTask(M, F, DirInfo.DirEnv, CloneEntryI, CloneExitI, FinalLoopInfo);
    } else if (IsWhile) {
      DirectiveWhileInfo FinalWhileInfo = DirInfo.DirEnv.WhileInfo;
      rewriteDirInfoForFinal(FinalWhileInfo, FinalInfo);
      NewCloneEntryI = buildWhileForTask(M, F, CloneEntryI, CloneExitI, FinalWhileInfo);
    }

    BasicBlock *OrigEntryBB = OrigEntryI->getParent();
    BasicBlock *OrigExitBB = OrigExitI->getParent()->getUniqueSuccessor();

    OrigExitBB->setName("final.end");
    BasicBlock *FinalCondBB = BasicBlock::Create(M.getContext(), "final.cond", &F);

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
    Value *Cond = IRB.CreateICmpNE(IRB.CreateCall(TaskInFinalFuncCallee, {}), IRB.getInt32(0));
    IRB.CreateCondBr(Cond, NewCloneEntryBB, OrigEntryBB);

  }

  void buildFinalCodes(
      Module &M, Function &F, DirectiveFunctionInfo &DirectiveFuncInfo) {

    // TODO: Avoid using a fixed array
    ValueToValueMapTy FinalInfo[500];
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
      ValueToValueMapTy &VMap = FinalInfo[i];

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
    }

    for (size_t i = 0; i < DirectiveFuncInfo.PostOrder.size(); ++i) {
      DirectiveInfo &DirInfo = *DirectiveFuncInfo.PostOrder[i];
      const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
      if (DirEnv.isOmpSsTaskDirective())
        buildFinalCondCFG(M, F, DirInfo, FinalInfo[i]);
    }

    // Erase cloned intrinsics
    for (size_t i = 0; i < DirectiveFuncInfo.PostOrder.size(); ++i) {
      DirectiveInfo &DirInfo = *DirectiveFuncInfo.PostOrder[i];
      const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
      if (DirEnv.isOmpSsTaskDirective()) {
        Instruction *OrigEntryI = DirInfo.Entry;
        Instruction *OrigExitI = DirInfo.Exit;
        Instruction *CloneEntryI = cast<Instruction>(FinalInfo[i].lookup(OrigEntryI));
        Instruction *CloneExitI = cast<Instruction>(FinalInfo[i].lookup(OrigExitI));

        bool IsDeviceWithNdrange =
          !DirEnv.DeviceInfo.empty()
            && !DirEnv.DeviceInfo.Ndrange.empty();
        if (IsDeviceWithNdrange)
          lowerTaskImpl(DirInfo, F, 666, M, CloneEntryI, CloneExitI);

        CloneExitI->eraseFromParent();
        CloneEntryI->eraseFromParent();
        for (size_t j = 0; j < DirInfo.InnerDirectiveInfos.size(); ++j) {
          DirectiveInfo &InnerDirInfo = *DirInfo.InnerDirectiveInfos[j];

          OrigEntryI = InnerDirInfo.Entry;
          OrigExitI = InnerDirInfo.Exit;
          CloneEntryI = cast<Instruction>(FinalInfo[i].lookup(OrigEntryI));
          CloneExitI = cast<Instruction>(FinalInfo[i].lookup(OrigExitI));

          bool IsDeviceWithNdrange =
            !InnerDirInfo.DirEnv.DeviceInfo.empty()
              && !InnerDirInfo.DirEnv.DeviceInfo.Ndrange.empty();
          if (IsDeviceWithNdrange) {
            llvm::Value *TmpIf = InnerDirInfo.DirEnv.If;
            InnerDirInfo.DirEnv.If = ConstantInt::getFalse(M.getContext());
            lowerTaskImpl(InnerDirInfo, F, 666, M, CloneEntryI, CloneExitI);
            InnerDirInfo.DirEnv.If = TmpIf;
          }

          CloneExitI->eraseFromParent();
          CloneEntryI->eraseFromParent();
        }
      }
    }
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
    olCallToUnpack(M, DirInfo, TaskArgsTy, TaskArgsToStructIdxMap, OlDestroyArgsFuncVar, UnpackDestroyArgsFuncVar, nullptr);

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
    Type *TaskAddrTranslationEntryTy = Nanos6TaskAddrTranslationEntry::getInstance(M).getType();
    TaskExtraTypeList.push_back(
      TaskAddrTranslationEntryTy->getPointerTo());
    TaskExtraNameList.push_back("address_translation_table");

    // CodeExtractor will create a entry block for us
    UnpackFunc = createUnpackOlFunction(
      M, F, ("nanos6_unpacked_task_region_" + F.getName() + Twine(taskNum)).str(),
      TaskTypeList, TaskNameList, TaskExtraTypeList, TaskExtraNameList, /*IsTask=*/true);

    OlFunc = createUnpackOlFunction(
      M, F, ("nanos6_ol_task_region_" + F.getName() + Twine(taskNum)).str(),
      {TaskArgsTy->getPointerTo()}, {"task_args"}, TaskExtraTypeList, TaskExtraNameList);

    olCallToUnpack(M, DirInfo, TaskArgsTy, TaskArgsToStructIdxMap, OlFunc, UnpackFunc, TaskAddrTranslationEntryTy, /*IsTaskFunc=*/true);
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
    olCallToUnpack(M, DirInfo, TaskArgsTy, TaskArgsToStructIdxMap, OlDepsFuncVar, UnpackDepsFuncVar, nullptr);

    return OlDepsFuncVar;
  }

  Function *createConstraintsOlFunc(
      Module &M, Function &F, int taskNum,
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, const DirectiveInfo &DirInfo,
      ArrayRef<Type *> TaskTypeList, ArrayRef<StringRef> TaskNameList) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveCostInfo &CostInfo = DirEnv.CostInfo;

    if (!CostInfo.Fun)
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
    unpackCostAndRewrite(M, CostInfo, UnpackConstraintsFuncVar, TaskArgsToStructIdxMap);

    Function *OlConstraintsFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_ol_constraints_" + F.getName() + Twine(taskNum)).str(),
                               {TaskArgsTy->getPointerTo()}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);
    olCallToUnpack(M, DirInfo, TaskArgsTy, TaskArgsToStructIdxMap, OlConstraintsFuncVar, UnpackConstraintsFuncVar, nullptr);

    return OlConstraintsFuncVar;
  }

  // Checks if the LoopInfo[i] depends on other iterator
  // NOTE: this assumes compute_lb/ub/step have an iterator as
  // an arguments if and only if it is used.
  bool isLoopIteratorDepenent(
      const DirectiveLoopInfo &LoopInfo, unsigned i) {
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

  bool multidepUsesLoopIter(
      const DirectiveLoopInfo &LoopInfo, const MultiDependInfo& MultiDepInfo) {
    for (const auto *V : MultiDepInfo.Args) {
      for (size_t i = 0; i < LoopInfo.IndVar.size(); ++i)
        if (V == LoopInfo.IndVar[i])
          return true;
    }
    return false;
  }

  bool hasMultidepUsingLoopIter(
      const DirectiveLoopInfo &LoopInfo, const DirectiveDependsInfo &DependsInfo) {
    for (auto &DepInfo : DependsInfo.List) {
      if (const auto *MultiDepInfo = dyn_cast<MultiDependInfo>(DepInfo.get())) {
        if (multidepUsesLoopIter(LoopInfo, *MultiDepInfo))
          return true;
      }
    }
    return false;
  }

  Function *createPriorityOlFunc(
      Module &M, Function &F, int taskNum,
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, const DirectiveInfo &DirInfo,
      ArrayRef<Type *> TaskTypeList, ArrayRef<StringRef> TaskNameList) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectivePriorityInfo &PriorityInfo = DirEnv.PriorityInfo;

    if (!PriorityInfo.Fun)
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
    unpackPriorityAndRewrite(M, PriorityInfo, UnpackPriorityFuncVar, TaskArgsToStructIdxMap);

    Function *OlPriorityFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_ol_priority_" + F.getName() + Twine(taskNum)).str(),
                               {TaskArgsTy->getPointerTo()}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);
    olCallToUnpack(M, DirInfo, TaskArgsTy, TaskArgsToStructIdxMap, OlPriorityFuncVar, UnpackPriorityFuncVar, nullptr);

    return OlPriorityFuncVar;
  }

  Function *createOnreadyOlFunc(
      Module &M, Function &F, int taskNum,
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, const DirectiveInfo &DirInfo,
      ArrayRef<Type *> TaskTypeList, ArrayRef<StringRef> TaskNameList) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveOnreadyInfo &OnreadyInfo = DirEnv.OnreadyInfo;

    if (!OnreadyInfo.Fun)
      return nullptr;

    Function *UnpackOnreadyFuncVar = createUnpackOlFunction(M, F,
                               ("nanos6_unpacked_onready_" + F.getName() + Twine(taskNum)).str(),
                               TaskTypeList, TaskNameList, {}, {});
    unpackOnreadyAndRewrite(M, OnreadyInfo, UnpackOnreadyFuncVar, TaskArgsToStructIdxMap);

    Function *OlOnreadyFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_ol_onready_" + F.getName() + Twine(taskNum)).str(),
                               {TaskArgsTy->getPointerTo()}, {"task_args"},
                               {}, {});
    olCallToUnpack(M, DirInfo, TaskArgsTy, TaskArgsToStructIdxMap, OlOnreadyFuncVar, UnpackOnreadyFuncVar, nullptr);

    return OlOnreadyFuncVar;
  }

  Function *createTaskIterWhileCondOlFunc(
      Module &M, Function &F, int taskNum,
      const MapVector<Value *, size_t> &TaskArgsToStructIdxMap,
      StructType *TaskArgsTy, const DirectiveInfo &DirInfo,
      ArrayRef<Type *> TaskTypeList, ArrayRef<StringRef> TaskNameList) {

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveWhileInfo &WhileInfo = DirEnv.WhileInfo;

    if (WhileInfo.empty())
      return nullptr;

    SmallVector<Type *, 4> TaskExtraTypeList;
    SmallVector<StringRef, 4> TaskExtraNameList;

    // uint8_t *result
    TaskExtraTypeList.push_back(Type::getInt8Ty(M.getContext())->getPointerTo());
    TaskExtraNameList.push_back("result");

    Function *UnpackWhileFuncVar = createUnpackOlFunction(M, F,
                               ("nanos6_unpacked_while_cond_" + F.getName() + Twine(taskNum)).str(),
                               TaskTypeList, TaskNameList,
                               TaskExtraTypeList, TaskExtraNameList);
    unpackWhileAndRewrite(M, WhileInfo, UnpackWhileFuncVar, TaskArgsToStructIdxMap);

    Function *OlWhileFuncVar
      = createUnpackOlFunction(M, F,
                               ("nanos6_ol_while_cond_" + F.getName() + Twine(taskNum)).str(),
                               {TaskArgsTy->getPointerTo()}, {"task_args"},
                               TaskExtraTypeList, TaskExtraNameList);
    olCallToUnpack(M, DirInfo, TaskArgsTy, TaskArgsToStructIdxMap, OlWhileFuncVar, UnpackWhileFuncVar, nullptr);

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
  Value *computeTaskFlags(IRBuilder<> &IRB, const DirectiveEnvironment &DirEnv) {
    const DirectiveLoopInfo &LoopInfo = DirEnv.LoopInfo;
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
    if (DirEnv.isOmpSsTaskIterDirective()) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
              ConstantInt::get(IRB.getInt64Ty(), 1),
              7));
    }
    if (LoopInfo.Update) {
      TaskFlagsVar =
        IRB.CreateOr(
          TaskFlagsVar,
          IRB.CreateShl(
            IRB.CreateZExt(
              LoopInfo.Update,
              IRB.getInt64Ty()),
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
      const DirectiveInfo &DirInfo, Module &M,
      Function &F, DirectiveLoopInfo &NewLoopInfo,
      Instruction *& NewEntryI, Instruction *& NewExitI,
      SmallVectorImpl<Value *> &NormalizedUBs,
      SmallVectorImpl<Instruction *> &CollapseStuff) {
    const DirectiveLoopInfo &LoopInfo = DirInfo.DirEnv.LoopInfo;

    SmallVector<Instruction *, 4> Allocas;
    gatherConstAllocas(Allocas, DirInfo.Entry);
    // Move the allocas before the loop start
    {
      IRBuilder<> IRB(DirInfo.Entry);
      for (Instruction *I : Allocas) {
        I->removeFromParent();
        IRB.Insert(I, I->getName());
      }
    }

    // This is used for nanos6_create_loop
    // NOTE: all values have nanos6 upper_bound type
    ComputeLoopBounds(M, LoopInfo, DirInfo.Entry, NormalizedUBs);

    IRBuilder<> IRB(DirInfo.Entry);

    Type *IndVarTy = DirInfo.DirEnv.getDSAType(LoopInfo.IndVar[0]);
    // Non collapsed loops build a loop using the original type
    NewLoopInfo.LBoundSigned[0] = LoopInfo.IndVarSigned[0];
    NewLoopInfo.UBoundSigned[0] = LoopInfo.IndVarSigned[0];
    NewLoopInfo.StepSigned[0] = LoopInfo.IndVarSigned[0];
    if (LoopInfo.LBound.size() > 1) {
      // Collapsed loops build a loop using size_t type to avoid overflows
      IndVarTy = IRB.getInt64Ty();
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
    buildLoopForTaskImpl(M, F, IndVarTy, DirInfo.Entry, DirInfo.Exit, NewLoopInfo, NewEntryI, NewExitI, CollapseStuff);
  }

  void lowerTaskImpl(
      const DirectiveInfo &DirInfo, Function &F,
      size_t taskNum, Module &M,
      Instruction *CloneEntry, Instruction *CloneExit) {

    Instruction *Entry = DirInfo.Entry;
    Instruction *Exit = DirInfo.Exit;
    if (CloneEntry && CloneExit) {
      Entry = CloneEntry;
      Exit = CloneExit;
    }

    const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
    const DirectiveLoopInfo &LoopInfo = DirInfo.DirEnv.LoopInfo;

    DebugLoc DLoc = Entry->getDebugLoc();
    unsigned Line = DLoc.getLine();
    unsigned Col = DLoc.getCol();
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
    StructType *TaskArgsTy = createTaskArgsType(M, DirInfo, TaskArgsToStructIdxMap,
                                                 ("nanos6_task_args_" + F.getName() + Twine(taskNum)).str());
    // Create nanos6_task_args_* END

    SmallVector<Type *, 4> TaskTypeList;
    SmallVector<StringRef, 4> TaskNameList;
    GlobalVariable *SizeofTableVar = nullptr;
    GlobalVariable *OffsetTableVar = nullptr;
    GlobalVariable *ArgIdxTableVar = nullptr;

    getTaskArgsInfo(
      DirEnv, M, F, TaskArgsToStructIdxMap, taskNum, TaskArgsTy, TaskTypeList, TaskNameList,
      SizeofTableVar, OffsetTableVar, ArgIdxTableVar);
   
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

    Function *OlOnreadyFuncVar =
      createOnreadyOlFunc(M, F, taskNum, TaskArgsToStructIdxMap, TaskArgsTy, DirInfo, TaskTypeList, TaskNameList);

    Function *OlTaskIterWhileCondFuncVar =
      createTaskIterWhileCondOlFunc(M, F, taskNum, TaskArgsToStructIdxMap, TaskArgsTy, DirInfo, TaskTypeList, TaskNameList);

    DirectiveLoopInfo NewLoopInfo = LoopInfo;
    SmallVector<Value *> NormalizedUBs(LoopInfo.UBound.size());
    SmallVector<Instruction *> CollapseStuff;
    if (DirEnv.isOmpSsLoopDirective())
      buildUnpackedLoopForTask(DirInfo, M, F, NewLoopInfo, NewEntryI, NewExitI, NormalizedUBs, CollapseStuff);
    if (DirEnv.isOmpSsTaskIterForDirective()) {
      Type *OrigIndVarTy = DirInfo.DirEnv.getDSAType(LoopInfo.IndVar[0]);
      // Increment induction variable at the end of the taskiter for
      IRBuilder<> IRB(DirInfo.Exit);
      Instruction *IndVarVal = IRB.CreateLoad(OrigIndVarTy, LoopInfo.IndVar[0]);
      Instruction *StepResult = IRB.CreateCall(LoopInfo.Step[0].Fun, LoopInfo.Step[0].Args);
      auto p = buildInstructionSignDependent(
        IRB, M, IndVarVal, StepResult, OrigIndVarTy, LoopInfo.StepSigned[0],
        [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
          return IRB.CreateAdd(LHS, RHS);
        });
      IRB.CreateStore(createZSExtOrTrunc(IRB, p.first, OrigIndVarTy, p.second), LoopInfo.IndVar[0]);
    }

    SetVector<Instruction *> TaskBBs;
    computeBBsBetweenEntryExit(TaskBBs, NewEntryI, NewExitI);

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

    GlobalVariable *TaskImplInfoVar =
      cast<GlobalVariable>(M.getOrInsertGlobal(
        ("implementations_var_" + F.getName() + Twine(taskNum)).str(),
        ArrayType::get(Nanos6TaskImplInfo::getInstance(M).getType(), 1)));
    TaskImplInfoVar->setLinkage(GlobalVariable::InternalLinkage);
    TaskImplInfoVar->setAlignment(Align(64));
    TaskImplInfoVar->setConstant(true);
    TaskImplInfoVar->setInitializer(
      ConstantArray::get(ArrayType::get(Nanos6TaskImplInfo::getInstance(M).getType(), 1), // TODO: More than one implementations?
        ConstantStruct::get(Nanos6TaskImplInfo::getInstance(M).getType(),
        DirEnv.DeviceInfo.Kind
          ? cast<Constant>(DirEnv.DeviceInfo.Kind)
          : ConstantInt::get(Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(0), 0),
        ConstantExpr::getPointerCast(OlTaskFuncVar, Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(1)),
        OlConstraintsFuncVar
          ? ConstantExpr::getPointerCast(OlConstraintsFuncVar,
                                         Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(2))
          : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(2))),
        DirEnv.Label
          ? ConstantExpr::getPointerCast(cast<Constant>(DirEnv.Label),
                                         Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(3))
          : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskImplInfo::getInstance(M).getType()->getElementType(3))),
        Nanos6TaskDeclSourceStr ? Nanos6TaskDeclSourceStr : Nanos6TaskLocStr,
        // Set device_function_name only in case of task pure device
        // in order to let nanos6 identify them
        (Nanos6TaskDevFuncStr && !DirEnv.DeviceInfo.Ndrange.empty())
          ? Nanos6TaskDevFuncStr
          : ConstantPointerNull::get(Type::getInt8PtrTy(M.getContext())))));


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
      OlOnreadyFuncVar
        ? ConstantExpr::getPointerCast(OlOnreadyFuncVar,
                                       Nanos6TaskInfo::getInstance(M).getType()->getElementType(2))
        : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(2))),
      OlPriorityFuncVar
        ? ConstantExpr::getPointerCast(OlPriorityFuncVar,
                                       Nanos6TaskInfo::getInstance(M).getType()->getElementType(3))
        : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(3))),
      ConstantInt::get(Nanos6TaskInfo::getInstance(M).getType()->getElementType(4), 1),
      ConstantExpr::getPointerCast(TaskImplInfoVar, Nanos6TaskInfo::getInstance(M).getType()->getElementType(5)),
      OlDestroyArgsFuncVar
        ? ConstantExpr::getPointerCast(OlDestroyArgsFuncVar,
                                       Nanos6TaskInfo::getInstance(M).getType()->getElementType(6))
        : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(6))),
      OlDuplicateArgsFuncVar
        ? ConstantExpr::getPointerCast(OlDuplicateArgsFuncVar,
                                       Nanos6TaskInfo::getInstance(M).getType()->getElementType(7))
        : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(7))),
      ConstantExpr::getPointerCast(TaskRedInitsVar, cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(8))),
      ConstantExpr::getPointerCast(TaskRedCombsVar, cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(9))),
      ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(10))),
      OlTaskIterWhileCondFuncVar
        ? ConstantExpr::getPointerCast(OlTaskIterWhileCondFuncVar,
                                       Nanos6TaskInfo::getInstance(M).getType()->getElementType(11))
        : ConstantPointerNull::get(cast<PointerType>(Nanos6TaskInfo::getInstance(M).getType()->getElementType(11))),
      ConstantInt::get(Nanos6TaskInfo::getInstance(M).getType()->getElementType(12), TaskTypeList.size()),
      ConstantExpr::getPointerCast(SizeofTableVar,
                                   Nanos6TaskInfo::getInstance(M).getType()->getElementType(13)),
      ConstantExpr::getPointerCast(OffsetTableVar,
                                   Nanos6TaskInfo::getInstance(M).getType()->getElementType(14)),
      ConstantExpr::getPointerCast(ArgIdxTableVar,
                                   Nanos6TaskInfo::getInstance(M).getType()->getElementType(15))));

    registerTaskInfo(M, TaskInfoVar);

    auto rewriteUsesBrAndGetOmpSsUnpackFunc
      = [&M, &LoopInfo, &DirInfo, &NewLoopInfo,
         &NormalizedUBs, &CollapseStuff, &UnpackTaskFuncVar, &TaskArgsToStructIdxMap,
         this, Entry, Exit]
           (BasicBlock *header, BasicBlock *newRootNode, BasicBlock *newHeader,
            Function *oldFunction, const SetVector<BasicBlock *> &Blocks) {
      const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
      UnpackTaskFuncVar->getBasicBlockList().push_back(newRootNode);

      if (DirEnv.isOmpSsLoopDirective()) {
        Type *OrigIndVarTy = DirInfo.DirEnv.getDSAType(LoopInfo.IndVar[0]);
        Type *NewIndVarTy = OrigIndVarTy;
        // Collapsed loops build a loop using size_t type to avoid overflows
        if (LoopInfo.LBound.size() > 1)
          NewIndVarTy = Type::getInt64Ty(M.getContext());

        IRBuilder<> IRB(&header->front());
        Value *LoopBounds = &*(UnpackTaskFuncVar->arg_end() - 2);

        Value *Idx[2];
        Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
        Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
        Value *LBoundField =
            IRB.CreateGEP(Nanos6LoopBounds::getInstance(M).getType(),
                          LoopBounds, Idx, "lb_gep");
        LBoundField = IRB.CreateLoad(
            Nanos6LoopBounds::getInstance(M).getType()->getElementType(1), LBoundField);
        LBoundField = IRB.CreateZExtOrTrunc(LBoundField, NewIndVarTy, "lb");

        Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 1);
        Value *UBoundField =
            IRB.CreateGEP(Nanos6LoopBounds::getInstance(M).getType(),
                          LoopBounds, Idx, "ub_gep");
        UBoundField = IRB.CreateLoad(
            Nanos6LoopBounds::getInstance(M).getType()->getElementType(2), UBoundField);
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
          Instruction *TmpEntry = &Entry->getParent()->front();
          TmpEntry = CollapseStuff[i];

          IRBuilder<> LoopBodyIRB(TmpEntry);
          if (!i)
            NormVal = LoopBodyIRB.CreateLoad(
                NewIndVarTy,
                NewLoopInfo.IndVar[0]);

          // NOTE: NormalizedUBs values are nanos6_register_loop upper_bound type
          Value *Niters = ConstantInt::get(Nanos6LoopBounds::getInstance(M).getType()->getElementType(1), 1);
          for (size_t j = i + 1; j < LoopInfo.LBound.size(); ++j)
            Niters = LoopBodyIRB.CreateMul(Niters, NormalizedUBs[j]);

          // TMP = LoopIndVar/ProductUBs(i+1..n)
          auto pTmp = buildInstructionSignDependent(
            LoopBodyIRB, M, NormVal, Niters, NewLoopInfo.IndVarSigned[0], /*RHSSigned=*/false,
            [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
              if (NewOpSigned)
                return IRB.CreateSDiv(LHS, RHS);
              return IRB.CreateUDiv(LHS, RHS);
            });

          // TMP * Step
          auto p = buildInstructionSignDependent(
            LoopBodyIRB, M, pTmp.first, LoopInfo.Step[i].Result, pTmp.second, LoopInfo.StepSigned[i],
            [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
              return IRB.CreateMul(LHS, RHS);
            });

          // (TMP * Step) + OrigLBound
          auto pBodyIndVar = buildInstructionSignDependent(
            LoopBodyIRB, M, p.first, LoopInfo.LBound[i].Result, p.second, LoopInfo.LBoundSigned[i],
            [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
              return IRB.CreateAdd(LHS, RHS);
            });

          // TMP*ProductUBs(i+1..n)
          p = buildInstructionSignDependent(
            LoopBodyIRB, M, pTmp.first, Niters, pTmp.second, /*RHSSigned=*/false,
            [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
              return IRB.CreateMul(LHS, RHS);
            });

          // LoopInfo - TMP*ProductUBs(i+1..n)
          auto pLoopIndVar = buildInstructionSignDependent(
            LoopBodyIRB, M, NormVal, p.first, NewLoopInfo.IndVarSigned[0], p.second,
            [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
              return IRB.CreateSub(LHS, RHS);
            });

          // LoopIndVar = LoopInfo - TMP*ProductUBs(i+1..n)
          NormVal = pLoopIndVar.first;

          // BodyIndVar(i) = (TMP * Step) + OrigLBound
          LoopBodyIRB.CreateStore(createZSExtOrTrunc(LoopBodyIRB, pBodyIndVar.first, OrigIndVarTy, pBodyIndVar.second), LoopInfo.IndVar[i]);

          if (isLoopIteratorDepenent(LoopInfo, i)) {
            // Create a check to skip unwanted iterations due to collapse guess
            Instruction *UBoundResult = LoopBodyIRB.CreateCall(LoopInfo.UBound[i].Fun, LoopInfo.UBound[i].Args);

            Instruction *IndVarVal = LoopBodyIRB.CreateLoad(
                DirInfo.DirEnv.getDSAType(LoopInfo.IndVar[i]),
                LoopInfo.IndVar[i]);
            Value *LoopCmp = nullptr;
            switch (LoopInfo.LoopType[i]) {
            case DirectiveLoopInfo::LT:
              LoopCmp = buildInstructionSignDependent(
                LoopBodyIRB, M, IndVarVal, UBoundResult, LoopInfo.IndVarSigned[i], LoopInfo.UBoundSigned[i],
                [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
                  if (NewOpSigned)
                    return IRB.CreateICmpSLT(LHS, RHS);
                  return IRB.CreateICmpULT(LHS, RHS);
                }).first;
              break;
            case DirectiveLoopInfo::LE:
              LoopCmp = buildInstructionSignDependent(
                LoopBodyIRB, M, IndVarVal, UBoundResult, LoopInfo.IndVarSigned[i], LoopInfo.UBoundSigned[i],
                [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
                  if (NewOpSigned)
                    return IRB.CreateICmpSLE(LHS, RHS);
                  return IRB.CreateICmpULE(LHS, RHS);
                }).first;
              break;
            case DirectiveLoopInfo::GT:
              LoopCmp = buildInstructionSignDependent(
                LoopBodyIRB, M, IndVarVal, UBoundResult, LoopInfo.IndVarSigned[i], LoopInfo.UBoundSigned[i],
                [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
                  if (NewOpSigned)
                    return IRB.CreateICmpSGT(LHS, RHS);
                  return IRB.CreateICmpUGT(LHS, RHS);
                }).first;
              break;
            case DirectiveLoopInfo::GE:
              LoopCmp = buildInstructionSignDependent(
                LoopBodyIRB, M, IndVarVal, UBoundResult, LoopInfo.IndVarSigned[i], LoopInfo.UBoundSigned[i],
                [](IRBuilder<> &IRB, Value *LHS, Value *RHS, bool NewOpSigned) {
                  if (NewOpSigned)
                    return IRB.CreateICmpSGE(LHS, RHS);
                  return IRB.CreateICmpUGE(LHS, RHS);
                }).first;
              break;
            default:
              llvm_unreachable("unexpected loop type");
            }

            // The IncrBB is the successor of BodyBB
            BasicBlock *BodyBB = Exit->getParent();
            Instruction *IncrBBI = &BodyBB->getUniqueSuccessor()->front();

            // Next iterator computation or BodyBB
            BasicBlock *NextBB = CollapseStuff[i]->getParent()->getUniqueSuccessor();

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
    auto emitOmpSsCaptureAndSubmitTask
      = [this, &M, &F, &DLoc, &TaskArgsTy,
         &DirEnv, &TaskArgsToStructIdxMap,
         &TaskInfoVar, &TaskInvInfoVar](
           Function *newFunction, BasicBlock *codeReplacer,
           const SetVector<BasicBlock *> &Blocks) {

      const DirectiveDSAInfo &DSAInfo = DirEnv.DSAInfo;
      const DirectiveVLADimsInfo &VLADimsInfo = DirEnv.VLADimsInfo;
      const DirectiveCapturedInfo &CapturedInfo = DirEnv.CapturedInfo;
      const DirectiveDependsInfo &DependsInfo = DirEnv.DependsInfo;
      const DirectiveLoopInfo &LoopInfo = DirEnv.LoopInfo;
      const DirectiveDeviceInfo &DeviceInfo = DirEnv.DeviceInfo;

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

      AllocaInst *TaskArgsVar = IRB.CreateAlloca(PointerType::getUnqual(M.getContext()));
      PostMoveInstructions.push_back(TaskArgsVar);
      Value *TaskFlagsVar = computeTaskFlags(IRB, DirEnv);
      AllocaInst *TaskPtrVar = IRB.CreateAlloca(PointerType::getUnqual(M.getContext()));
      PostMoveInstructions.push_back(TaskPtrVar);

      Value *TaskArgsStructSizeOf = IRB.getInt64(M.getDataLayout().getTypeAllocSize(TaskArgsTy));

      // TODO: this forces an alignment of 16 for VLAs
      {
        const int ALIGN = 16;
        TaskArgsStructSizeOf =
          IRB.CreateNUWAdd(TaskArgsStructSizeOf, IRB.getInt64(ALIGN - 1));
        TaskArgsStructSizeOf =
          IRB.CreateAnd(TaskArgsStructSizeOf, IRB.CreateNot(IRB.getInt64(ALIGN - 1)));
      }

      Value *TaskArgsVLAsExtraSizeOf = computeTaskArgsVLAsExtraSizeOf(M, DirEnv, IRB, VLADimsInfo);
      Value *TaskArgsSizeOf = IRB.CreateNUWAdd(TaskArgsStructSizeOf, TaskArgsVLAsExtraSizeOf);

      Type *NumDependenciesTy = IRB.getInt64Ty();
      Instruction *NumDependencies = IRB.CreateAlloca(NumDependenciesTy, nullptr, "num.deps");
      PostMoveInstructions.push_back(NumDependencies);

      if (DirEnv.isOmpSsTaskLoopDirective() && hasMultidepUsingLoopIter(LoopInfo, DependsInfo)) {
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

            buildLoopForMultiDep(
                M, F, DirEnv, NumDependenciesLoad, NumDependenciesStore,
                MultiDepInfo);
          }
        }
      }

      // Arguments for creating a task or a loop directive
      SmallVector<Value *, 4> CreateDirectiveArgs = {
          TaskInfoVar,
          TaskInvInfoVar,
          DirEnv.InstanceLabel
            ? IRB.CreateBitCast(DirEnv.InstanceLabel, IRB.getInt8PtrTy())
            : Constant::getNullValue(IRB.getInt8PtrTy()),
          TaskArgsSizeOf,
          TaskArgsVar,
          TaskPtrVar,
          TaskFlagsVar,
          IRB.CreateLoad(NumDependenciesTy, NumDependencies)
      };

      if (DirEnv.isOmpSsLoopDirective()) {
        SmallVector<Value *> NormalizedUBs(LoopInfo.UBound.size());
        ComputeLoopBounds(M, LoopInfo, &*IRB.saveIP().getPoint(), NormalizedUBs);

        Value *Niters = ConstantInt::get(Nanos6LoopBounds::getInstance(M).getType()->getElementType(1), 1);
        for (size_t i = 0; i < LoopInfo.LBound.size(); ++i)
          Niters = IRB.CreateMul(Niters, NormalizedUBs[i]);

        Value *RegisterGrainsize =
          ConstantInt::get(
            Nanos6LoopBounds::getInstance(M).getType()->getElementType(2), 0);
        if (DirEnv.LoopInfo.Grainsize)
          RegisterGrainsize = DirEnv.LoopInfo.Grainsize;

        Value *RegisterChunksize =
          ConstantInt::get(
            Nanos6LoopBounds::getInstance(M).getType()->getElementType(3), 0);
        if (DirEnv.LoopInfo.Chunksize)
          RegisterChunksize = DirEnv.LoopInfo.Chunksize;

        Value *RegisterLowerB = ConstantInt::get(Nanos6LoopBounds::getInstance(M).getType()->getElementType(0), 0);
        CreateDirectiveArgs.push_back(RegisterLowerB);
        CreateDirectiveArgs.push_back(Niters);
        CreateDirectiveArgs.push_back(
          createZSExtOrTrunc(
            IRB, RegisterGrainsize,
            Nanos6LoopBounds::getInstance(M).getType()->getElementType(2), /*Signed=*/false));
        CreateDirectiveArgs.push_back(
          createZSExtOrTrunc(
            IRB, RegisterChunksize,
            Nanos6LoopBounds::getInstance(M).getType()->getElementType(3), /*Signed=*/false));

        IRB.CreateCall(CreateLoopFuncCallee, CreateDirectiveArgs);
      } else if (DirEnv.isOmpSsTaskIterDirective()) {
        Value *Niters = ConstantInt::get(Nanos6LoopBounds::getInstance(M).getType()->getElementType(1), 1);
        if (DirEnv.isOmpSsTaskIterForDirective()) {
          SmallVector<Value *> NormalizedUBs(LoopInfo.UBound.size());
          ComputeLoopBounds(M, LoopInfo, &*IRB.saveIP().getPoint(), NormalizedUBs);

          for (size_t i = 0; i < LoopInfo.LBound.size(); ++i)
            Niters = IRB.CreateMul(Niters, NormalizedUBs[i]);
        }

        Value *RegisterUnroll =
          ConstantInt::get(
            CreateIterFuncCallee.getFunctionType()->getParamType(9), 0);
        if (LoopInfo.Unroll)
          RegisterUnroll = LoopInfo.Unroll;

        Value *RegisterLowerB = ConstantInt::get(Nanos6LoopBounds::getInstance(M).getType()->getElementType(0), 0);
        CreateDirectiveArgs.push_back(RegisterLowerB);
        CreateDirectiveArgs.push_back(Niters);
        CreateDirectiveArgs.push_back(
          createZSExtOrTrunc(
            IRB, RegisterUnroll,
            CreateIterFuncCallee.getFunctionType()->getParamType(9), /*Signed=*/false));

        IRB.CreateCall(CreateIterFuncCallee, CreateDirectiveArgs);
      } else {
        IRB.CreateCall(CreateTaskFuncCallee, CreateDirectiveArgs);
      }

      // DSA capture
      Value *TaskArgsVarL = IRB.CreateLoad(
          PointerType::getUnqual(M.getContext()), TaskArgsVar);

      Value *TaskArgsVarLi8IdxGEP =
        IRB.CreateGEP(IRB.getInt8Ty(), TaskArgsVarL, TaskArgsStructSizeOf, "args_end");

      SmallVector<VLAAlign, 2> VLAAlignsInfo;
      computeVLAsAlignOrder(M, VLAAlignsInfo, DirEnv, VLADimsInfo);

      // TODO: is this a good place to put this?
      if (!DeviceInfo.empty()) {
        int DevGEPIdx = 0;
        // Add device info
        Value *Idx[2], *GEP;
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
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
            Idx[1] = ConstantInt::get(IRB.getInt32Ty(), DevGEPIdx++);
            GEP = IRB.CreateGEP(
                TaskArgsTy,
                TaskArgsVarL, Idx, ("gep_dev_ndrange" + Twine(i)).str());
            Value *V = ConstantInt::get(IRB.getInt64Ty(), -1);
            if (j <= DeviceInfo.HasLocalSize && i < NdrangeLength)
              V = createZSExtOrTrunc(
                IRB, DeviceInfo.Ndrange[NdrangeLength*j + i],
                IRB.getInt64Ty(), /*Signed=*/false);

            IRB.CreateStore(V, GEP);
          }
        }
        // size_t shm_size;
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), DevGEPIdx++);
        GEP = IRB.CreateGEP(
            TaskArgsTy,
            TaskArgsVarL, Idx, "gep_dev_shm");
        IRB.CreateStore(
          ConstantInt::get(IRB.getInt64Ty(), 0), GEP);
      }

      // First point VLAs to its according space in task args
      for (const auto& VAlign : VLAAlignsInfo) {
        auto *V = VAlign.V;
        Type *Ty = DirEnv.getDSAType(V);
        unsigned TyAlign = VAlign.Align;

        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsToStructIdxMap[V]);
        Value *GEP =
            IRB.CreateGEP(TaskArgsTy,
                          TaskArgsVarL, Idx, "gep_" + V->getName());

        // Point VLA in task args to an aligned position of the extra space allocated
        IRB.CreateAlignedStore(TaskArgsVarLi8IdxGEP, GEP, Align(TyAlign));
        // Skip current VLA size
        unsigned SizeB = M.getDataLayout().getTypeAllocSize(Ty);
        Value *VLASize = ConstantInt::get(IRB.getInt64Ty(), SizeB);
        for (auto *Dim : VLADimsInfo.lookup(V))
          VLASize = IRB.CreateNUWMul(VLASize, Dim);
        TaskArgsVarLi8IdxGEP = IRB.CreateGEP(
            IRB.getInt8Ty(),
            TaskArgsVarLi8IdxGEP, VLASize);
      }

      for (Value *V : DSAInfo.Shared) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsToStructIdxMap[V]);
        Value *GEP = IRB.CreateGEP(
            TaskArgsTy,
            TaskArgsVarL, Idx, "gep_" + V->getName());
        IRB.CreateStore(V, GEP);
      }
      for (size_t i = 0; i < DSAInfo.Private.size(); ++i) {
        Value *V = DSAInfo.Private[i];
        Type *Ty = DSAInfo.PrivateTy[i];
        // Call custom constructor generated in clang in non-pods
        // Leave pods unititialized
        auto It = DirEnv.NonPODsInfo.Inits.find(V);
        if (It != DirEnv.NonPODsInfo.Inits.end()) {
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
              TaskArgsTy,
              TaskArgsVarL, Idx, "gep_" + V->getName());

          // VLAs
          if (VLADimsInfo.count(V))
            GEP = IRB.CreateLoad(PointerType::getUnqual(M.getContext()), GEP);

          IRB.CreateCall(FunctionCallee(cast<Function>(It->second)), ArrayRef<Value*>{GEP, NSize});
        }
      }
      for (size_t i = 0; i < DSAInfo.Firstprivate.size(); ++i) {
        Value *V = DSAInfo.Firstprivate[i];
        Type *Ty = DSAInfo.FirstprivateTy[i];
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
            TaskArgsTy,
            TaskArgsVarL, Idx, "gep_" + V->getName());

        // VLAs
        if (VLADimsInfo.count(V))
          GEP = IRB.CreateLoad(PointerType::getUnqual(M.getContext()), GEP);

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
            TaskArgsTy,
            TaskArgsVarL, Idx, "capt_gep_" + V->getName());
        IRB.CreateStore(V, GEP);
      }

      Value *TaskPtrVarL = IRB.CreateLoad(
          IRB.getPtrTy(), TaskPtrVar);

      CallInst *TaskSubmitFuncCall = IRB.CreateCall(TaskSubmitFuncCallee, TaskPtrVarL);
      return TaskSubmitFuncCall;
    };
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
    CodeExtractor CE(TaskBBs1, rewriteUsesBrAndGetOmpSsUnpackFunc, emitOmpSsCaptureAndSubmitTask, valueInUnpackParams);
    CE.extractCodeRegion(CEAC);

    // FIXME: CollapseStuff should be local to task
    CollapseStuff.clear();
  }

  void lowerTask(
      const DirectiveInfo &DirInfo, Function &F,
      size_t taskNum, Module &M) {

    Instruction *CloneEntry = nullptr;
    Instruction *CloneExit = nullptr;

    lowerTaskImpl(
      DirInfo, F, taskNum, M, CloneEntry, CloneEntry);

    DirInfo.Exit->eraseFromParent();
    DirInfo.Entry->eraseFromParent();
  }

  void buildNanos6Types(Module &M) {
    // void nanos6_create_task(
    //         nanos6_task_info_t *task_info,
    //         nanos6_task_invocation_info_t *task_invocation_info,
    //         const char *task_label,
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
        Type::getInt8PtrTy(M.getContext()),
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

    // void nanos6_create_loop(
    //     nanos6_task_info_t *task_info,
    //     nanos6_task_invocation_info_t *task_invocation_info,
    //     const char *task_label,
    //     size_t args_block_size,
    //     /* OUT */ void **args_block_pointer,
    //     /* OUT */ void **task_pointer,
    //     size_t flags,
    //     size_t num_deps,
    //     size_t lower_bound,
    //     size_t upper_bound,
    //     size_t grainsize,
    //     size_t chunksize
    // );
    CreateLoopFuncCallee = M.getOrInsertFunction("nanos6_create_loop",
        Type::getVoidTy(M.getContext()),
        Nanos6TaskInfo::getInstance(M).getType()->getPointerTo(),
        Nanos6TaskInvInfo::getInstance(M).getType()->getPointerTo(),
        Type::getInt8PtrTy(M.getContext()),
        Type::getInt64Ty(M.getContext()),
        Type::getInt8PtrTy(M.getContext())->getPointerTo(),
        Type::getInt8PtrTy(M.getContext())->getPointerTo(),
        Type::getInt64Ty(M.getContext()),
        Type::getInt64Ty(M.getContext()),
        Type::getInt64Ty(M.getContext()),
        Type::getInt64Ty(M.getContext()),
        Type::getInt64Ty(M.getContext()),
        Type::getInt64Ty(M.getContext())
    );

    // void nanos6_create_iter(
    //     nanos6_task_info_t *task_info,
    //     nanos6_task_invocation_info_t *task_invocation_info,
    //     const char *task_label,
    //     size_t args_block_size,
    //     /* OUT */ void **args_block_pointer,
    //     /* OUT */ void **task_pointer,
    //     size_t flags,
    //     size_t num_deps,
    //     size_t lower_bound,
    //     size_t upper_bound,
    //     size_t unroll
    // );
    CreateIterFuncCallee = M.getOrInsertFunction("nanos6_create_iter",
        Type::getVoidTy(M.getContext()),
        Nanos6TaskInfo::getInstance(M).getType()->getPointerTo(),
        Nanos6TaskInvInfo::getInstance(M).getType()->getPointerTo(),
        Type::getInt8PtrTy(M.getContext()),
        Type::getInt64Ty(M.getContext()),
        Type::getInt8PtrTy(M.getContext())->getPointerTo(),
        Type::getInt8PtrTy(M.getContext())->getPointerTo(),
        Type::getInt64Ty(M.getContext()),
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
    {
      cast<Function>(TaskInfoRegisterCtorFuncCallee.getCallee())->setLinkage(GlobalValue::InternalLinkage);
      BasicBlock *EntryBB = BasicBlock::Create(M.getContext(), "entry",
        cast<Function>(TaskInfoRegisterCtorFuncCallee.getCallee()));
      EntryBB->getInstList().push_back(ReturnInst::Create(M.getContext()));

      appendToGlobalCtors(M, cast<Function>(TaskInfoRegisterCtorFuncCallee.getCallee()), 65535);
    }

    // void nanos6_config_assert(const char *str);
    RegisterAssertFuncCallee =
      M.getOrInsertFunction("nanos6_config_assert",
        Type::getVoidTy(M.getContext()),
        Type::getInt8PtrTy(M.getContext())
      );

    // void nanos6_constructor_register_assert(void);
    // NOTE: This does not belong to nanos6 API
    RegisterCtorAssertFuncCallee =
      M.getOrInsertFunction("nanos6_constructor_register_assert",
        Type::getVoidTy(M.getContext())
      );
    {
      cast<Function>(RegisterCtorAssertFuncCallee.getCallee())->setLinkage(GlobalValue::InternalLinkage);
      BasicBlock *EntryBB = BasicBlock::Create(M.getContext(), "entry",
        cast<Function>(RegisterCtorAssertFuncCallee.getCallee()));
      EntryBB->getInstList().push_back(ReturnInst::Create(M.getContext()));

      appendToGlobalCtors(M, cast<Function>(RegisterCtorAssertFuncCallee.getCallee()), 65535);

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
                registerAssert(M, ValueStrRef);
              }
            }
          }
        }
      }
    }
  }

  void relocateInstrs() {
    for (auto *I : PostMoveInstructions) {
      Function *TmpF = nullptr;
      for (User *U : I->users()) {
        if (Instruction *I2 = dyn_cast<Instruction>(U)) {
          Function *DstF = cast<Function>(I2->getParent()->getParent());
          assert((!TmpF || TmpF == DstF) &&
            "Instruction I has uses in differents functions");
          TmpF = DstF;
          Instruction *TI = DstF->getEntryBlock().getTerminator();
          I->moveBefore(TI);
        }
      }
    }
  }

  bool runOnModule(Module &M) {
    if (M.empty())
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
      DirectiveFunctionInfo &DirectiveFuncInfo = LookupDirectiveFunctionInfo(*F).getFuncInfo();

      buildFinalCodes(M, *F, DirectiveFuncInfo);
      size_t taskNum = 0;
      for (size_t i = 0; i < DirectiveFuncInfo.PostOrder.size(); ++i) {
        DirectiveInfo &DirInfo = *DirectiveFuncInfo.PostOrder[i];
        const DirectiveEnvironment &DirEnv = DirInfo.DirEnv;
        if (DirEnv.isOmpSsTaskwaitDirective())
          lowerTaskwait(DirInfo, M);
        else if (DirEnv.isOmpSsReleaseDirective())
          lowerRelease(DirInfo, M);
        else if (DirEnv.isOmpSsTaskDirective())
          lowerTask(DirInfo, *F, taskNum++, M);
      }

    }
    relocateInstrs();
    return true;
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
    return OmpSs(LookupDirectiveFunctionInfo).runOnModule(M);
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
  if (!OmpSs(LookupDirectiveFunctionInfo).runOnModule(M))
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
