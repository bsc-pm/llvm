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
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
using namespace llvm;

namespace {

struct OmpSs : public ModulePass {
  /// Pass identification, replacement for typeid
  static char ID;
  OmpSs() : ModulePass(ID) {
    initializeOmpSsPass(*PassRegistry::getPassRegistry());
  }

  bool Initialized = false;

  struct TaskAddrTranslationEntryTy {
    StructType *Ty;
  };
  TaskAddrTranslationEntryTy TskAddrTranslationEntryTy;

  struct TaskConstraintsTy {
    StructType *Ty;
  };
  TaskConstraintsTy TskConstraintsTy;

  struct TaskInvInfoTy {
    struct Members {
      Type *InvSourceTy;
    };
    StructType *Ty;
    Members Mmbers;
  };
  TaskInvInfoTy TskInvInfoTy;

  struct TaskImplInfoTy {
    struct Members {
      Type *DeviceTypeIdTy;
      Type *RunFuncTy;
      Type *GetConstraintsFuncTy;
      Type *TaskLabelTy;
      Type *DeclSourceTy;
      Type *RunWrapperFuncTy;
    };
    StructType *Ty;
    Members Mmbers;
  };
  TaskImplInfoTy TskImplInfoTy;

  struct TaskInfoTy {
    struct Members {
        Type *NumSymbolsTy;
        Type *RegisterInfoFuncTy;
        Type *GetPriorityFuncTy;
        Type *TypeIdTy;
        Type *ImplCountTy;
        Type *TskImplInfoTy;
        Type *DestroyArgsBlockFuncTy;
        Type *DuplicateArgsBlockFuncTy;
        Type *ReductInitsFuncTy;
        Type *ReductCombsFuncTy;
    };
    StructType *Ty;
    Members Mmbers;
  };
  TaskInfoTy TskInfoTy;

  FunctionCallee CreateTaskFuncTy;
  FunctionCallee TaskSubmitFuncTy;
  enum {
    DEP_IN,
    DEP_OUT,
    DEP_INOUT,
    DEP_WEAKIN,
    DEP_WEAKOUT,
    DEP_WEAKINOUT,
    DEP_ENUM_SIZE,
  };

  static const int MAX_DEP_DIMS = 8;
  // TODO: Use Map and a function to look up if it exists already
  SmallVector<SmallVector<FunctionCallee, DEP_ENUM_SIZE>, MAX_DEP_DIMS> RegisterRegionsTypes;

  void unpackDepsOfType(const SmallVectorImpl<DependInfo> &DependList, Function *F) {
    BasicBlock &Entry = F->getEntryBlock();
    for (const DependInfo &DI : DependList) {
      for (Instruction * const &I : DI.UnpackInstructions) {
        I->removeFromParent();
        Entry.getInstList().push_back(I);
      }
    }
  }

  void unpackDeps(const TaskDependsInfo &TDI, Function *F, ArrayRef<Value *> DSAMerge) {
    unpackDepsOfType(TDI.Ins, F);
    unpackDepsOfType(TDI.Outs, F);
    unpackDepsOfType(TDI.Inouts, F);
    unpackDepsOfType(TDI.WeakIns, F);
    unpackDepsOfType(TDI.WeakOuts, F);
    unpackDepsOfType(TDI.WeakInouts, F);
    // Rewrite instructions to use arguments
    for (Instruction &I : F->getEntryBlock()) {
      Function::arg_iterator AI = F->arg_begin();
      for (unsigned i = 0, e = DSAMerge.size(); i != e; ++i, ++AI) {
        I.replaceUsesOfWith(DSAMerge[i], &*AI);
      }
    }
  }

  void unpackCallToRTOfType(Module &M,
                            const SmallVectorImpl<DependInfo> &DependList,
                            Function *F,
                            ArrayRef<Value *> DSAMerge,
                            int DepType) {
    for (const DependInfo &DI : DependList) {
      IRBuilder<> BBBuilder(&F->getEntryBlock().back());

      // Dependency Base rewrite:
      //   GlobalValue
      //     replace by task argument
      //   DSA
      //     replace by task argument
      //   No DSA
      //     do nothing since it's in unpack instructions
      Value *NewBase = DI.Base;
      bool Found = false;
      Function::arg_iterator AI = F->arg_begin();
      for (unsigned i = 0, e = DSAMerge.size(); i != e && !Found; ++i) {
        if (DSAMerge[i] == NewBase)
          Found = true;
        else
          ++AI;
      }
      if (Found)
        NewBase = &*AI;

      Value *BaseCast = BBBuilder.CreateBitCast(NewBase, Type::getInt8PtrTy(M.getContext()));
      SmallVector<Value *, 4> TaskDepAPICall;
      Value *Handler = &*(F->arg_end() - 1);
      TaskDepAPICall.push_back(Handler);
      TaskDepAPICall.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()), DI.SymbolIndex));
      TaskDepAPICall.push_back(ConstantPointerNull::get(Type::getInt8PtrTy(M.getContext()))); // TODO: stringify
      TaskDepAPICall.push_back(BaseCast);
      assert(!(DI.Dims.size()%3));
      int NumDims = DI.Dims.size()/3 - 1;
      for (Value *V : DI.Dims) {
        TaskDepAPICall.push_back(V);
      }
      BBBuilder.CreateCall(RegisterRegionsTypes[NumDims][DepType], TaskDepAPICall);
    }
  }

  void unpackDepsCallToRT(Module &M,
                      const TaskDependsInfo &TDI,
                      Function *F,
                      ArrayRef<Value *> DSAMerge) {
    unpackCallToRTOfType(M, TDI.Ins, F, DSAMerge, DEP_IN);
    unpackCallToRTOfType(M, TDI.Outs, F, DSAMerge, DEP_OUT);
    unpackCallToRTOfType(M, TDI.Inouts, F, DSAMerge, DEP_INOUT);
    unpackCallToRTOfType(M, TDI.WeakIns, F, DSAMerge, DEP_WEAKIN);
    unpackCallToRTOfType(M, TDI.WeakOuts, F, DSAMerge, DEP_WEAKOUT);
    unpackCallToRTOfType(M, TDI.WeakInouts, F, DSAMerge, DEP_WEAKINOUT);
  }

  // Creates an empty UnpackDeps Function with entry BB.
  Function *createUnpackDepsFunction(Module &M, Function &F, std::string Suffix, ArrayRef<Value *> DSAMerge) {
    Type *RetTy = Type::getVoidTy(M.getContext());
    std::vector<Type *> ParamsTy;
    for (Value *V : DSAMerge) {
      ParamsTy.push_back(V->getType());
    }
    ParamsTy.push_back(Type::getInt8PtrTy(M.getContext())); /* void * handler */
    FunctionType *UnpackDepsFuncType =
      FunctionType::get(RetTy, ParamsTy, /*IsVarArgs */ false);

    Function *UnpackDepsFuncVar = Function::Create(
        UnpackDepsFuncType, GlobalValue::InternalLinkage, F.getAddressSpace(),
        "nanos6_unpacked_deps_" + F.getName() + Suffix, &M);

    BasicBlock::Create(M.getContext(), "entry", UnpackDepsFuncVar);
    return UnpackDepsFuncVar;
  }

  // Creates an empty UnpackTask Function without entry BB.
  // CodeExtractor will create it for us
  Function *createUnpackTaskFunction(Module &M, Function &F, std::string Suffix,
                                     ArrayRef<Value *> DSAMerge, SetVector<BasicBlock *> &TaskBBs,
                                     Type *TaskAddrTranslationEntryTy) {
    Type *RetTy = Type::getVoidTy(M.getContext());
    std::vector<Type *> ParamsTy;
    for (Value *V : DSAMerge) {
      ParamsTy.push_back(V->getType());
    }
    ParamsTy.push_back(Type::getInt8PtrTy(M.getContext())); /* void * device_env */
    ParamsTy.push_back(TaskAddrTranslationEntryTy->getPointerTo()); /* nanos6_address_translation_entry_t *address_translation_table */
    FunctionType *UnpackTaskFuncType =
      FunctionType::get(RetTy, ParamsTy, /*IsVarArgs */ false);

    Function *UnpackTaskFuncVar = Function::Create(
        UnpackTaskFuncType, GlobalValue::InternalLinkage, F.getAddressSpace(),
        "nanos6_unpacked_task_region_" + F.getName() + Suffix, &M);

    // Create an iterator to name all of the arguments we inserted.
    Function::arg_iterator AI = UnpackTaskFuncVar->arg_begin();
    // Rewrite all users of the DSAMerge in the extracted region to use the
    // arguments (or appropriate addressing into struct) instead.
    for (unsigned i = 0, e = DSAMerge.size(); i != e; ++i) {
      Value *RewriteVal = &*AI++;

      std::vector<User *> Users(DSAMerge[i]->user_begin(), DSAMerge[i]->user_end());
      for (User *use : Users)
        if (Instruction *inst = dyn_cast<Instruction>(use))
          if (TaskBBs.count(inst->getParent()))
            inst->replaceUsesOfWith(DSAMerge[i], RewriteVal);
    }
    // Set names for arguments.
    AI = UnpackTaskFuncVar->arg_begin();
    for (unsigned i = 0, e = DSAMerge.size(); i != e; ++i, ++AI)
      AI->setName(DSAMerge[i]->getName());

    return UnpackTaskFuncVar;
  }

  // Create an OutlineDeps Function with entry BB
  Function *createOlDepsFunction(Module &M, Function &F, std::string Suffix, Type *TaskArgsTy) {
    Type *RetTy = Type::getVoidTy(M.getContext());
    std::vector<Type *> ParamsTy;
    ParamsTy.push_back(TaskArgsTy->getPointerTo());
    ParamsTy.push_back(Type::getInt8PtrTy(M.getContext())); /* void * handler */
    FunctionType *OlDepsFuncType =
                    FunctionType::get(RetTy, ParamsTy, /*IsVarArgs */ false);

    Function *OlDepsFuncVar = Function::Create(
        OlDepsFuncType, GlobalValue::InternalLinkage, F.getAddressSpace(),
        "nanos6_ol_deps_" + F.getName() + Suffix, &M);

    BasicBlock::Create(M.getContext(), "entry", OlDepsFuncVar);
    return OlDepsFuncVar;
  }

  // Create an OutlineTask Function with entry BB
  Function *createOlTaskFunction(Module &M, Function &F, std::string Suffix, Type *TaskArgsTy,
                                 Type *TaskAddrTranslationEntryTy) {
    Type *RetTy = Type::getVoidTy(M.getContext());
    std::vector<Type *> ParamsTy;
    ParamsTy.push_back(TaskArgsTy->getPointerTo());
    ParamsTy.push_back(Type::getInt8PtrTy(M.getContext())); /* void * device_env */
    ParamsTy.push_back(TaskAddrTranslationEntryTy->getPointerTo()); /* nanos6_address_translation_entry_t *address_translation_table */
    FunctionType *OlTaskFuncType =
                    FunctionType::get(RetTy, ParamsTy, /*IsVarArgs */ false);

    Function *OlTaskFuncVar = Function::Create(
        OlTaskFuncType, GlobalValue::InternalLinkage, F.getAddressSpace(),
        "nanos6_ol_task_region_" + F.getName() + Suffix, &M);

    BasicBlock::Create(M.getContext(), "entry", OlTaskFuncVar);
    return OlTaskFuncVar;
  }

  // Given a Outline Function assuming that task args are the first parameter, and
  // DSAInfo, it unpacks task args in Outline and fills UnpackedList with those Values,
  // used to call Unpack Functions
  void unpackDSAs(Module &M, const TaskDSAInfo &DSAInfo, Function *OlFunc,
                  SmallVectorImpl<Value *> &UnpackedList) {
    UnpackedList.clear();

    IRBuilder<> BBBuilder(&OlFunc->getEntryBlock());
    Function::arg_iterator AI = OlFunc->arg_begin();
    Value *OlDepsFuncTaskArgs = &*AI++;
    unsigned TaskArgsIdx = 0;
    for (unsigned i = 0; i < DSAInfo.Shared.size(); ++i, ++TaskArgsIdx) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), TaskArgsIdx);
      Value *GEP = BBBuilder.CreateGEP(
          OlDepsFuncTaskArgs, Idx, "gep_" + DSAInfo.Shared[i]->getName());
      Value *LGEP = BBBuilder.CreateLoad(GEP, "load_" + GEP->getName());
      UnpackedList.push_back(LGEP);
    }
    for (unsigned i = 0; i < DSAInfo.Private.size(); ++i, ++TaskArgsIdx) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), TaskArgsIdx);
      Value *GEP = BBBuilder.CreateGEP(
          OlDepsFuncTaskArgs, Idx, "gep_" + DSAInfo.Private[i]->getName());
      UnpackedList.push_back(GEP);
    }
    for (unsigned i = 0; i < DSAInfo.Firstprivate.size(); ++i, ++TaskArgsIdx) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), TaskArgsIdx);
      Value *GEP = BBBuilder.CreateGEP(
          OlDepsFuncTaskArgs, Idx, "gep_" + DSAInfo.Firstprivate[i]->getName());
      UnpackedList.push_back(GEP);
    }
  }

  // Given an OutlineDeps and UnpackDeps Functions it unpacks DSAs in Outline
  // and builds a call to Unpack
  void olDepsCallToUnpack(Module &M, const TaskDSAInfo &DSAInfo,
                          Function *OlFunc, Function *UnpackFunc) {
    IRBuilder<> BBBuilder(&OlFunc->getEntryBlock());

    // First arg is the nanos_task_args
    Function::arg_iterator AI = OlFunc->arg_begin();
    AI++;
    SmallVector<Value *, 4> TaskDepsUnpackParams;
    unpackDSAs(M, DSAInfo, OlFunc, TaskDepsUnpackParams);
    TaskDepsUnpackParams.push_back(&*AI++);
    // Build TaskUnpackCall
    BBBuilder.CreateCall(UnpackFunc, TaskDepsUnpackParams);
    // Make BB legal with a terminator to task outline function
    BBBuilder.CreateRetVoid();
  }

  // Given an OutlineTask and UnpackTask Functions it unpacks DSAs in Outline
  // and builds a call to Unpack
  void olTaskCallToUnpack(Module &M, const TaskDSAInfo &DSAInfo,
                          Function *OlFunc, Function *UnpackFunc) {
    IRBuilder<> BBBuilder(&OlFunc->getEntryBlock());

    // First arg is the nanos_task_args
    Function::arg_iterator AI = OlFunc->arg_begin();
    AI++;
    SmallVector<Value *, 4> TaskUnpackParams;
    unpackDSAs(M, DSAInfo, OlFunc, TaskUnpackParams);
    TaskUnpackParams.push_back(&*AI++);
    TaskUnpackParams.push_back(&*AI++);
    // Build TaskUnpackCall
    BBBuilder.CreateCall(UnpackFunc, TaskUnpackParams);
    // Make BB legal with a terminator to task outline function
    BBBuilder.CreateRetVoid();
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

  void lowerTask(const TaskInfo &TI,
                 Function &F,
                 size_t taskNum,
                 Module &M) {


    unsigned Line = TI.Entry->getDebugLoc().getLine();
    unsigned Col = TI.Entry->getDebugLoc().getCol();
    std::string FileNamePlusLoc = (M.getSourceFileName()
                                   + ":" + Twine(Line)
                                   + ":" + Twine(Col)).str();

    Constant *Nanos6TaskLocStr = IRBuilder<>(TI.Entry).CreateGlobalStringPtr(FileNamePlusLoc);

    // 1. Split BB
    BasicBlock *EntryBB = TI.Entry->getParent();
    EntryBB = EntryBB->splitBasicBlock(TI.Entry);

    BasicBlock *ExitBB = TI.Exit->getParent();
    ExitBB = ExitBB->splitBasicBlock(TI.Exit);

    // 2. Gather BB between entry and exit (is there any function/util to do this?)
    ReversePostOrderTraversal<BasicBlock *> RPOT(EntryBB);
    SetVector<BasicBlock *> TaskBBs;
    for (BasicBlock *BB : RPOT) {
      // End of task reached, done
      if (BB == ExitBB)
        break;
      TaskBBs.insert(BB);
    }

    // Create nanos6_task_args_* START
    // Private and Firstprivate must be stored in the struct
    SmallVector<Type *, 4> TaskArgsMemberTy;
    for (Value *V : TI.DSAInfo.Shared) {
      TaskArgsMemberTy.push_back(V->getType());
    }
    for (Value *V : TI.DSAInfo.Private) {
      TaskArgsMemberTy.push_back(V->getType()->getPointerElementType());
    }
    for (Value *V : TI.DSAInfo.Firstprivate) {
      TaskArgsMemberTy.push_back(V->getType()->getPointerElementType());
    }
    StructType *TaskArgsTy = StructType::create(M.getContext(), TaskArgsMemberTy, ("nanos6_task_args_" + F.getName() + Twine(taskNum)).str());
    // Create nanos6_task_args_* END

    SetVector<Value *> DSAMerge;
    DSAMerge.insert(TI.DSAInfo.Shared.begin(), TI.DSAInfo.Shared.end());
    DSAMerge.insert(TI.DSAInfo.Private.begin(), TI.DSAInfo.Private.end());
    DSAMerge.insert(TI.DSAInfo.Firstprivate.begin(), TI.DSAInfo.Firstprivate.end());

    // nanos6_unpacked_task_region_* START
    Function *UnpackTaskFuncVar
      = createUnpackTaskFunction(M, F, Twine(taskNum).str(),
                                 DSAMerge.getArrayRef(), TaskBBs,
                                 TskAddrTranslationEntryTy.Ty);

    // nanos6_unpacked_task_region_* END

    // nanos6_ol_task_region_* START
    Function *OlTaskFuncVar
      = createOlTaskFunction(M, F, Twine(taskNum).str(), TaskArgsTy, TskAddrTranslationEntryTy.Ty);

    olTaskCallToUnpack(M, TI.DSAInfo, OlTaskFuncVar, UnpackTaskFuncVar);

    // nanos6_ol_task_region_* END

    // nanos6_unpacked_deps_* START

    Function *UnpackDepsFuncVar
      = createUnpackDepsFunction(M, F, Twine(taskNum).str(), DSAMerge.getArrayRef());

    unpackDeps(TI.DependsInfo, UnpackDepsFuncVar, DSAMerge.getArrayRef());
    UnpackDepsFuncVar->getEntryBlock().getInstList().push_back(ReturnInst::Create(M.getContext()));

    unpackDepsCallToRT(M, TI.DependsInfo, UnpackDepsFuncVar, DSAMerge.getArrayRef());

    // nanos6_unpacked_deps_* END

    // nanos6_ol_deps_* START

    Function *OlDepsFuncVar
      = createOlDepsFunction(M, F, Twine(taskNum).str(), TaskArgsTy);

    olDepsCallToUnpack(M, TI.DSAInfo, OlDepsFuncVar, UnpackDepsFuncVar);

    // nanos6_ol_deps_* END

    // 3. Create Nanos6 task data structures info
    Constant *TaskInvInfoVar = M.getOrInsertGlobal(("task_invocation_info_" + F.getName() + Twine(taskNum)).str(),
                                      TskInvInfoTy.Ty,
                                      [&] {
      GlobalVariable *GV = new GlobalVariable(M, TskInvInfoTy.Ty,
                                false,
                                GlobalVariable::InternalLinkage,
                                ConstantStruct::get(TskInvInfoTy.Ty,
                                                    Nanos6TaskLocStr),
                                ("task_invocation_info_" + F.getName() + Twine(taskNum)).str());
      GV->setAlignment(64);
      return GV;
    });

    Constant *TaskImplInfoVar = M.getOrInsertGlobal(("implementations_var_" + F.getName() + Twine(taskNum)).str(),
                                      ArrayType::get(TskImplInfoTy.Ty, 1),
                                      [&] {
      auto *OlTaskFuncCastTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                              {Type::getInt8PtrTy(M.getContext()), /* void * */
                                               Type::getInt8PtrTy(M.getContext()), /* void * */
                                               TskAddrTranslationEntryTy.Ty->getPointerTo()
                                               }, false);

      GlobalVariable *GV = new GlobalVariable(M, ArrayType::get(TskImplInfoTy.Ty, 1),
                                false,
                                GlobalVariable::InternalLinkage,
                                ConstantArray::get(ArrayType::get(TskImplInfoTy.Ty, 1), // TODO: More than one implementations?
                                                   ConstantStruct::get(TskImplInfoTy.Ty,
                                                                       ConstantInt::get(TskImplInfoTy.Mmbers.DeviceTypeIdTy, 0),
                                                                       ConstantExpr::getPointerCast(OlTaskFuncVar, OlTaskFuncCastTy->getPointerTo()),
                                                                       ConstantPointerNull::get(TskImplInfoTy.Mmbers.GetConstraintsFuncTy->getPointerTo()),
                                                                       ConstantPointerNull::get(cast<PointerType>(TskImplInfoTy.Mmbers.TaskLabelTy)), /* const char *task_label */
                                                                       Nanos6TaskLocStr,
                                                                       ConstantPointerNull::get(TskImplInfoTy.Mmbers.RunWrapperFuncTy->getPointerTo()))),
                                ("implementations_var_" + F.getName() + Twine(taskNum)).str());

      GV->setAlignment(64);
      return GV;
    });

    Constant *TaskInfoVar = M.getOrInsertGlobal(("task_info_var_" + F.getName() + Twine(taskNum)).str(),
                                      TskInfoTy.Ty,
                                      [&] {
      GlobalVariable *GV = new GlobalVariable(M, TskInfoTy.Ty,
                                false,
                                GlobalVariable::InternalLinkage,
                                ConstantStruct::get(TskInfoTy.Ty,
                                                    // TODO: Add support for devices
                                                    ConstantInt::get(TskInfoTy.Mmbers.NumSymbolsTy, TI.DependsInfo.NumSymbols),
                                                    ConstantExpr::getPointerCast(OlDepsFuncVar, TskInfoTy.Mmbers.RegisterInfoFuncTy->getPointerTo()),
                                                    ConstantPointerNull::get(TskInfoTy.Mmbers.GetPriorityFuncTy->getPointerTo()),
                                                    ConstantPointerNull::get(cast<PointerType>(TskInfoTy.Mmbers.TypeIdTy)),
                                                    ConstantInt::get(TskInfoTy.Mmbers.ImplCountTy, 1),
                                                    ConstantExpr::getPointerCast(TaskImplInfoVar, TskInfoTy.Mmbers.TskImplInfoTy->getPointerTo()),
                                                    ConstantPointerNull::get(TskInfoTy.Mmbers.DestroyArgsBlockFuncTy->getPointerTo()),
                                                    ConstantPointerNull::get(TskInfoTy.Mmbers.DuplicateArgsBlockFuncTy->getPointerTo()),
                                                    ConstantPointerNull::get(TskInfoTy.Mmbers.ReductInitsFuncTy->getPointerTo()),
                                                    ConstantPointerNull::get(TskInfoTy.Mmbers.ReductCombsFuncTy->getPointerTo())),
                                ("task_info_var_" + F.getName() + Twine(taskNum)).str());

      GV->setAlignment(64);
      return GV;
    });

    auto rewriteOutToInTaskBrAndGetOmpSsUnpackFunc = [&](BasicBlock *header,
                                              BasicBlock *newRootNode,
                                              BasicBlock *newHeader,
                                              Function *oldFunction,
                                              Module *M,
                                              const SetVector<BasicBlock *> &Blocks) {

      UnpackTaskFuncVar->getBasicBlockList().push_back(newRootNode);

      // Rewrite branches from basic blocks outside of the task region to blocks
      // inside the region to use the new label (newHeader) since the task region
      // will be outlined
      std::vector<User *> Users(header->user_begin(), header->user_end());
      for (unsigned i = 0, e = Users.size(); i != e; ++i)
        // The BasicBlock which contains the branch is not in the region
        // modify the branch target to a new block
        if (Instruction *I = dyn_cast<Instruction>(Users[i]))
          if (I->isTerminator() && !Blocks.count(I->getParent()) &&
              I->getParent()->getParent() == oldFunction)
            I->replaceUsesOfWith(header, newHeader);

      return UnpackTaskFuncVar;
    };
    auto emitOmpSsCaptureAndSubmitTask = [&](Function *newFunction,
                                  BasicBlock *codeReplacer,
                                  const SetVector<BasicBlock *> &Blocks) {

      IRBuilder<> IRB(codeReplacer);
      Value *TaskArgsVar = IRB.CreateAlloca(TaskArgsTy->getPointerTo());
      Value *TaskArgsVarCast = IRB.CreateBitCast(TaskArgsVar, IRB.getInt8PtrTy()->getPointerTo());
      // TODO: For now TaskFlagsVar is hardcoded
      // Value *TaskFlagsVar = IRB.CreateAlloca(IRB.getInt64Ty());
      // IRB.CreateStore(ConstantInt::get(IRB.getInt64Ty(), 0), TaskFlagsVar);
      Value *TaskPtrVar = IRB.CreateAlloca(IRB.getInt8PtrTy());
      // TODO: For now TaskNumDepsVar is hardcoded
      // Value *TaskNumDepsVar = IRB.CreateAlloca(IRB.getInt64Ty());
      // IRB.CreateStore(ConstantInt::get(IRB.getInt64Ty(), 0), TaskNumDepsVar);
      uint64_t TaskArgsSizeOf = M.getDataLayout().getTypeAllocSize(TaskArgsTy);
      IRB.CreateCall(CreateTaskFuncTy, {TaskInfoVar,
                                  TaskInvInfoVar,
                                  ConstantInt::get(IRB.getInt64Ty(), TaskArgsSizeOf),
                                  TaskArgsVarCast,
                                  TaskPtrVar,
                                  ConstantInt::get(IRB.getInt64Ty(), 0), // TaskFlagsVar,
                                  ConstantInt::get(IRB.getInt64Ty(),
                                                   TI.DependsInfo.Ins.size()
                                                   + TI.DependsInfo.Outs.size())}); // TaskNumDepsVar;

      // DSA capture
      Value *TaskArgsVarL = IRB.CreateLoad(TaskArgsVar);
      size_t TaskArgsIdx = 0;
      for (unsigned i = 0; i < TI.DSAInfo.Shared.size(); ++i, ++TaskArgsIdx) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsIdx);
        Value *GEP = IRB.CreateGEP(
            TaskArgsVarL, Idx, "gep_" + TI.DSAInfo.Shared[i]->getName());
        IRB.CreateStore(TI.DSAInfo.Shared[i], GEP);
      }
      TaskArgsIdx += TI.DSAInfo.Private.size();
      for (unsigned i = 0; i < TI.DSAInfo.Firstprivate.size(); ++i, ++TaskArgsIdx) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
        Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsIdx);
        Value *GEP = IRB.CreateGEP(
            TaskArgsVarL, Idx, "gep_" + TI.DSAInfo.Firstprivate[i]->getName());
        Value *FPValue = IRB.CreateLoad(TI.DSAInfo.Firstprivate[i]);
        IRB.CreateStore(FPValue, GEP);
      }

      Value *TaskPtrVarL = IRB.CreateLoad(TaskPtrVar);
      CallInst *TaskSubmitFuncCall = IRB.CreateCall(TaskSubmitFuncTy, TaskPtrVarL);

      // Add a branch to the next basic block after the task region
      // and replace the terminator that exits the task region
      // Since this is a single entry single exit region this should
      // be done once.
      bool DoneOnce = false;
      for (BasicBlock *Block : Blocks) {
        Instruction *TI = Block->getTerminator();
        for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
          if (!Blocks.count(TI->getSuccessor(i))) {
            assert(!DoneOnce && "More than one exit in task code");
            DoneOnce = true;

            BasicBlock *OldTarget = TI->getSuccessor(i);

            // Create branch to next BB after the task region
            IRB.CreateBr(OldTarget);

            IRBuilder<> BNewTerminatorI(TI);
            BNewTerminatorI.CreateRetVoid();
          }
          if (DoneOnce)
            TI->eraseFromParent();
      }

      return TaskSubmitFuncCall;
    };

    // 4. Extract region the way we want
    CodeExtractor CE(TaskBBs.getArrayRef(), rewriteOutToInTaskBrAndGetOmpSsUnpackFunc, emitOmpSsCaptureAndSubmitTask);
    CE.extractCodeRegion();

    TI.Exit->eraseFromParent();
    TI.Entry->eraseFromParent();
  }

  void buildNanos6Types(Module &M) {
    TskAddrTranslationEntryTy.Ty = StructType::create(M.getContext(), "nanos6_address_translation_entry_t");

    TskConstraintsTy.Ty = StructType::create(M.getContext(), "nanos6_task_constraints_t");

    TskInvInfoTy.Ty = StructType::create(M.getContext(), "nanos6_task_invocation_info_t");
    TskInvInfoTy.Mmbers.InvSourceTy = Type::getInt8PtrTy(M.getContext());
    TskInvInfoTy.Ty->setBody(TskInvInfoTy.Mmbers.InvSourceTy); /* const char *invocation_source */

    TskImplInfoTy.Ty = StructType::create(M.getContext(), "nanos6_task_implementation_info_t");
    TskImplInfoTy.Mmbers.DeviceTypeIdTy = Type::getInt32Ty(M.getContext());
    TskImplInfoTy.Mmbers.RunFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                   {Type::getInt8PtrTy(M.getContext()), /* void * */
                                    Type::getInt8PtrTy(M.getContext()), /* void * */
                                    TskAddrTranslationEntryTy.Ty->getPointerTo()},
                                   /*IsVarArgs=*/false);
    TskImplInfoTy.Mmbers.GetConstraintsFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                   {Type::getInt8PtrTy(M.getContext()), /* void * */
                                    TskConstraintsTy.Ty->getPointerTo()},
                                   /*IsVarArgs=*/false);
    TskImplInfoTy.Mmbers.TaskLabelTy = Type::getInt8PtrTy(M.getContext());
    TskImplInfoTy.Mmbers.DeclSourceTy = Type::getInt8PtrTy(M.getContext());
    TskImplInfoTy.Mmbers.RunWrapperFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                   {Type::getInt8PtrTy(M.getContext()), /* void * */
                                    Type::getInt8PtrTy(M.getContext()), /* void * */
                                    TskAddrTranslationEntryTy.Ty->getPointerTo()},
                                   /*IsVarArgs=*/false);
    TskImplInfoTy.Ty->setBody({TskImplInfoTy.Mmbers.DeviceTypeIdTy, /* int device_type_id */
                             TskImplInfoTy.Mmbers.RunFuncTy->getPointerTo(),
                             TskImplInfoTy.Mmbers.GetConstraintsFuncTy->getPointerTo(),
                             TskImplInfoTy.Mmbers.TaskLabelTy, /* const char *task_label */
                             TskImplInfoTy.Mmbers.DeclSourceTy, /* const char *declaration_source*/
                             TskImplInfoTy.Mmbers.RunWrapperFuncTy->getPointerTo()
                            });

    TskInfoTy.Ty = StructType::create(M.getContext(), "nanos6_task_info_t");
    TskInfoTy.Mmbers.NumSymbolsTy = Type::getInt32Ty(M.getContext());;
    TskInfoTy.Mmbers.RegisterInfoFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {Type::getInt8PtrTy(M.getContext()), /* void * */
                                        Type::getInt8PtrTy(M.getContext()), /* void * */
                                       },
                                       /*IsVarArgs=*/false);
    TskInfoTy.Mmbers.GetPriorityFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                   {Type::getInt8PtrTy(M.getContext()), /* void * */
                                    Type::getInt64PtrTy(M.getContext()), /* nanos6_priority_t = long int * */
                                   },
                                   /*IsVarArgs=*/false);
    TskInfoTy.Mmbers.TypeIdTy = Type::getInt8PtrTy(M.getContext());
    TskInfoTy.Mmbers.ImplCountTy = Type::getInt32Ty(M.getContext());
    TskInfoTy.Mmbers.TskImplInfoTy = TskImplInfoTy.Ty->getPointerTo();
    TskInfoTy.Mmbers.DestroyArgsBlockFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                   {Type::getInt8PtrTy(M.getContext()), /* void * */
                                   },
                                   /*IsVarArgs=*/false);
    TskInfoTy.Mmbers.DuplicateArgsBlockFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                   {Type::getInt8PtrTy(M.getContext()), /* const void * */
                                    Type::getInt8PtrTy(M.getContext())->getPointerTo(), /* void ** */
                                   },
                                   /*IsVarArgs=*/false);
    TskInfoTy.Mmbers.ReductInitsFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                   {Type::getInt8PtrTy(M.getContext()), /* void * */
                                    Type::getInt8PtrTy(M.getContext()), /* void * */
                                    Type::getInt64Ty(M.getContext()), /* size_t */
                                   },
                                   /*IsVarArgs=*/false);
    TskInfoTy.Mmbers.ReductCombsFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                   {Type::getInt8PtrTy(M.getContext()), /* void * */
                                    Type::getInt8PtrTy(M.getContext()), /* void * */
                                    Type::getInt64Ty(M.getContext()), /* size_t */
                                   },
                                   /*IsVarArgs=*/false);
    TskInfoTy.Ty->setBody({TskInfoTy.Mmbers.NumSymbolsTy,
                           TskInfoTy.Mmbers.RegisterInfoFuncTy->getPointerTo(),
                           TskInfoTy.Mmbers.GetPriorityFuncTy->getPointerTo(),
                           TskInfoTy.Mmbers.TypeIdTy, /* const char *type_identifier */
                           TskInfoTy.Mmbers.ImplCountTy, /* int implementation_count */
                           TskInfoTy.Mmbers.TskImplInfoTy->getPointerTo(),
                           TskInfoTy.Mmbers.DestroyArgsBlockFuncTy->getPointerTo(),
                           TskInfoTy.Mmbers.DuplicateArgsBlockFuncTy->getPointerTo(),
                           TskInfoTy.Mmbers.ReductInitsFuncTy->getPointerTo(),
                           TskInfoTy.Mmbers.ReductCombsFuncTy->getPointerTo(),
                          });

    // Create function types
    // nanos6_create_task
    // nanos6_submit_task
    // nanos6_register_region_write_depinfo1
    // nanos6_register_region_write_depinfo1

    CreateTaskFuncTy = M.getOrInsertFunction("nanos6_create_task",
        Type::getVoidTy(M.getContext()),
        TskInfoTy.Ty->getPointerTo(),                       // nanos6_task_info_t *task_info
        TskInvInfoTy.Ty->getPointerTo(),                    // nanos6_task_invocation_info_t *task_invocation_info
        Type::getInt64Ty(M.getContext()),                   // size_t args_lock_size
        Type::getInt8PtrTy(M.getContext())->getPointerTo(), // void **args_block_pointer
        Type::getInt8PtrTy(M.getContext())->getPointerTo(), // void **task_pointer
        Type::getInt64Ty(M.getContext()),                   // size_t flags
        Type::getInt64Ty(M.getContext())                    // size_t num_deps
    );

    TaskSubmitFuncTy = M.getOrInsertFunction("nanos6_submit_task",
        Type::getVoidTy(M.getContext()),
        Type::getInt8PtrTy(M.getContext()) // void *task
    );

    SmallVector<Type *, 8> RegisterRegionParams;
    static const char *DepFuncNames[] = {
      "read",
      "write",
      "readwrite",
      "weak_read",
      "weak_write",
      "weak_readwrite"
    };

    // void *handler, Task handler
    RegisterRegionParams.push_back(Type::getInt8PtrTy(M.getContext()));
    // int symbol_index, Argument identifier
    RegisterRegionParams.push_back(Type::getInt32Ty(M.getContext()));
    // char const *region_text, Stringified contents of the dependency clause
    RegisterRegionParams.push_back(Type::getInt8PtrTy(M.getContext()));
    // void *base_address
    RegisterRegionParams.push_back(Type::getInt8PtrTy(M.getContext()));
    for (int i = 0; i < MAX_DEP_DIMS; ++i) {
      // long dimsize
      RegisterRegionParams.push_back(Type::getInt64Ty(M.getContext()));
      // long dimstart
      RegisterRegionParams.push_back(Type::getInt64Ty(M.getContext()));
      // long dimend
      RegisterRegionParams.push_back(Type::getInt64Ty(M.getContext()));
      SmallVector<FunctionCallee, DEP_ENUM_SIZE> RegisterRegionsOfDim;
      for (int j = 0; j < DEP_ENUM_SIZE; ++j) {
        FunctionType *RegisterRegionFuncType =
                        FunctionType::get(Type::getVoidTy(M.getContext()),
                                          RegisterRegionParams,
                                          /*IsVarArgs */ false);
        RegisterRegionsOfDim.push_back(
          M.getOrInsertFunction(
            ("nanos6_register_region_" + Twine(DepFuncNames[j]) + "_depinfo" + Twine(i + 1)).str(),
            RegisterRegionFuncType));
      }
      RegisterRegionsTypes.push_back(RegisterRegionsOfDim);
    }

    // RegisterRegionRead1Ty = M.getOrInsertFunction("nanos6_register_region_read_depinfo1",
    //     Type::getVoidTy(M.getContext()),
    //     Type::getInt8PtrTy(M.getContext()),
    //     Type::getInt32Ty(M.getContext()),
    //     Type::getInt8PtrTy(M.getContext()),
    //     Type::getInt8PtrTy(M.getContext()),
    //     Type::getInt64Ty(M.getContext()),   // long dim1size
    //     Type::getInt64Ty(M.getContext()),   // long dim1start
    //     Type::getInt64Ty(M.getContext())    // long dim1end
    // );

    // RegisterRegionWrite1Ty = M.getOrInsertFunction("nanos6_register_region_write_depinfo1",
    //     Type::getVoidTy(M.getContext()),
    //     Type::getInt8PtrTy(M.getContext()), // void *handler, Task handler
    //     Type::getInt32Ty(M.getContext()),   // int symbol_index, Argument identifier
    //     Type::getInt8PtrTy(M.getContext()), // char const *region_text, Stringified contents of the dependency clause
    //     Type::getInt8PtrTy(M.getContext()), // void *base_address
    //     Type::getInt64Ty(M.getContext()),   // long dim1size
    //     Type::getInt64Ty(M.getContext()),   // long dim1start
    //     Type::getInt64Ty(M.getContext()));  // long dim1end
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

      for (TaskwaitInfo& TwI : TwFI.PostOrder) {
        lowerTaskwait(TwI, M);
      }
      size_t taskNum = 0;
      for (TaskInfo TI : TFI.PostOrder) {
        lowerTask(TI, *F, taskNum++, M);
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
