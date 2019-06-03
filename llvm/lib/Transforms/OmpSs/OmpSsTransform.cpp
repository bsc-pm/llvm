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

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    for (auto &F : M) {
      // Nothing to do for declarations.
      if (F.isDeclaration() || F.empty())
        continue;

      TaskFunctionInfo &TFI = getAnalysis<OmpSsRegionAnalysisPass>(F).getTaskFuncInfo();
      TaskwaitFunctionInfo &TwFI = getAnalysis<OmpSsRegionAnalysisPass>(F).getTaskwaitFuncInfo();

      for (TaskwaitInfo TwI : TwFI.PostOrder) {
        // 1. Create Taskwait function Type
        IRBuilder<> IRB(TwI.I);
        Function *Func = cast<Function>(M.getOrInsertFunction(
            "nanos6_taskwait", IRB.getVoidTy(), IRB.getInt8PtrTy()));
        // 2. Build String
        Constant *Nanos6TaskwaitStr = IRB.CreateGlobalStringPtr(M.getModuleIdentifier());

        // 3. Insert the call
        IRB.CreateCall(Func, {Nanos6TaskwaitStr});
        // 4. Remove the intrinsic
        TwI.I->eraseFromParent();
      }
      for (TaskInfo TI : TFI.PostOrder) {
        // 0. Create all nanos6 data structure types if not are defined yet
        StructType *TaskAddrTranslationEntryTy = StructType::create(M.getContext(), "nanos6_address_translation_entry_t");
        StructType *TaskConstraintsTy = StructType::create(M.getContext(), "nanos6_task_constraints_t");

        StructType *TaskInvInfoTy = StructType::create(M.getContext(), "nanos6_task_invocation_info_t");
        TaskInvInfoTy->setBody({Type::getInt8PtrTy(M.getContext())}); /* const char *invocation_source */

        StructType *TaskImplInfoTy = StructType::create(M.getContext(), "nanos6_task_implementation_info_t");
        auto *RunFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {Type::getInt8PtrTy(M.getContext()), /* void * */
                                        Type::getInt8PtrTy(M.getContext()), /* void * */
                                        TaskAddrTranslationEntryTy->getPointerTo()},
                                       /*IsVarArgs=*/false);
        auto *GetConstraintsFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {Type::getInt8PtrTy(M.getContext()), /* void * */
                                        TaskConstraintsTy->getPointerTo()},
                                       /*IsVarArgs=*/false);
        auto *RunWrapperFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {Type::getInt8PtrTy(M.getContext()), /* void * */
                                        Type::getInt8PtrTy(M.getContext()), /* void * */
                                        TaskAddrTranslationEntryTy->getPointerTo()},
                                       /*IsVarArgs=*/false);
        TaskImplInfoTy->setBody({Type::getInt32Ty(M.getContext()), /* int device_type_id */
                                 RunFuncTy->getPointerTo(),
                                 GetConstraintsFuncTy->getPointerTo(),
                                 Type::getInt8PtrTy(M.getContext()), /* const char *task_label */
                                 Type::getInt8PtrTy(M.getContext()), /* const char *declaration_source*/
                                 RunWrapperFuncTy->getPointerTo()
                                });
        StructType *TaskInfoTy = StructType::create(M.getContext(), "nanos6_task_info_t");
        auto *RegisterInfoFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {Type::getInt8PtrTy(M.getContext()), /* void * */
                                        Type::getInt8PtrTy(M.getContext()), /* void * */
                                       },
                                       /*IsVarArgs=*/false);
        auto *GetPriorityFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {Type::getInt8PtrTy(M.getContext()), /* void * */
                                        Type::getInt64PtrTy(M.getContext()), /* nanos6_priority_t = long int * */
                                       },
                                       /*IsVarArgs=*/false);
        auto *DestroyArgsBlockFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {Type::getInt8PtrTy(M.getContext()), /* void * */
                                       },
                                       /*IsVarArgs=*/false);
        auto *DuplicateArgsBlockFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {Type::getInt8PtrTy(M.getContext()), /* const void * */
                                        Type::getInt8PtrTy(M.getContext())->getPointerTo(), /* void ** */
                                       },
                                       /*IsVarArgs=*/false);
        auto *ReductInitsFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {Type::getInt8PtrTy(M.getContext()), /* void * */
                                        Type::getInt8PtrTy(M.getContext()), /* void * */
                                        Type::getInt64Ty(M.getContext()), /* size_t */
                                       },
                                       /*IsVarArgs=*/false);
        auto *ReductCombsFuncTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {Type::getInt8PtrTy(M.getContext()), /* void * */
                                        Type::getInt8PtrTy(M.getContext()), /* void * */
                                        Type::getInt64Ty(M.getContext()), /* size_t */
                                       },
                                       /*IsVarArgs=*/false);
        TaskInfoTy->setBody({Type::getInt32Ty(M.getContext()), /* int num_symbols */
                                 RegisterInfoFuncTy->getPointerTo(),
                                 GetPriorityFuncTy->getPointerTo(),
                                 Type::getInt8PtrTy(M.getContext()), /* const char *type_identifier */
                                 Type::getInt32Ty(M.getContext()), /* int implementation_count */
                                 TaskImplInfoTy->getPointerTo(),
                                 DestroyArgsBlockFuncTy->getPointerTo(),
                                 DuplicateArgsBlockFuncTy->getPointerTo(),
                                 ReductInitsFuncTy->getPointerTo(),
                                 ReductCombsFuncTy->getPointerTo(),
                                });

        // Create nanos6_task_args_*
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
        StructType *TaskArgsTy = StructType::create(M.getContext(), TaskArgsMemberTy, ("nanos6_task_args_" + F.getName()).str());


          // auto *NewLoad = new LoadInst(V, LI->getName() + ".pre",
          //                              LI->isVolatile(), LI->getAlignment(),
          //                              LI->getOrdering(), LI->getSyncScopeID(),
          //                              UnavailablePred->getTerminator());

        // 1. Split BB
        BasicBlock *EntryBB = TI.Entry->getParent();
        EntryBB = EntryBB->splitBasicBlock(TI.Entry);

        BasicBlock *ExitBB = TI.Exit->getParent();
        ExitBB = ExitBB->splitBasicBlock(TI.Exit);

        // 2. Gather BB between entry and exit (is there any function/util to do this?)
        ReversePostOrderTraversal<BasicBlock *> RPOT(EntryBB);
        SmallVector<BasicBlock *, 4> TaskBBs;
        for (BasicBlock *BB : RPOT) {
          // End of task reached, done
          if (BB == ExitBB)
            break;
          TaskBBs.push_back(BB);
        }

        #if 1
        TI.Exit->eraseFromParent();
        TI.Entry->eraseFromParent();
        #endif
        // CodeExtractor CE(TaskBBs,
        //                  /* DominatorTree */ nullptr,
        //                  /* AggregateArgs */ false,
        //                  /* BFI */ nullptr,
        //                  /* BPI */ nullptr,
        //                  /* AllowVarArgs */ false,
        //                  /* AllowAlloca */ true,
        //                  /* Suffix */ "task.");

        #if 1
        TaskDSAInfo &DSAInfo = TI.DSAInfo;

        Constant *TaskInvInfoVar;
        Constant *TaskImplInfoVar;
        Constant *TaskInfoVar;

        auto constructOmpSsFunctions = [&](BasicBlock *header,
                                                  BasicBlock *newRootNode,
                                                  BasicBlock *newHeader,
                                                  Function *oldFunction,
                                                  Module *M,
                                                  const SetVector<BasicBlock *> &Blocks) {

          Type *RetTy = Type::getVoidTy(M->getContext());
          std::vector<Type *> paramTy;
          SetVector<Value *> DSAMerge;
          DSAMerge.insert(TI.DSAInfo.Shared.begin(), TI.DSAInfo.Shared.end());
          DSAMerge.insert(TI.DSAInfo.Private.begin(), TI.DSAInfo.Private.end());
          DSAMerge.insert(TI.DSAInfo.Firstprivate.begin(), TI.DSAInfo.Firstprivate.end());
          // nanos6_unpacked_task_region_*
          for (Value *value : DSAMerge) {
            paramTy.push_back(value->getType());
          }
          paramTy.push_back(Type::getInt8PtrTy(M->getContext())); /* void * device_env */
          paramTy.push_back(TaskAddrTranslationEntryTy->getPointerTo()); /* nanos6_address_translation_entry_t *address_translation_table */
          FunctionType *unpackFuncType =
                          FunctionType::get(RetTy, paramTy, /*IsVarArgs */ false);

          Function *unpackFuncVar = Function::Create(
              unpackFuncType, GlobalValue::InternalLinkage, oldFunction->getAddressSpace(),
              "nanos6_unpacked_task_region_" + oldFunction->getName(), M);
          unpackFuncVar->getBasicBlockList().push_back(newRootNode);

          // Create an iterator to name all of the arguments we inserted.
          Function::arg_iterator AI = unpackFuncVar->arg_begin();
          // Rewrite all users of the DSAMerge in the extracted region to use the
          // arguments (or appropriate addressing into struct) instead.
          for (unsigned i = 0, e = DSAMerge.size(); i != e; ++i) {
            Value *RewriteVal = &*AI++;

            std::vector<User *> Users(DSAMerge[i]->user_begin(), DSAMerge[i]->user_end());
            for (User *use : Users)
              if (Instruction *inst = dyn_cast<Instruction>(use))
                if (Blocks.count(inst->getParent()))
                  inst->replaceUsesOfWith(DSAMerge[i], RewriteVal);
          }
          // Set names for arguments.
          AI = unpackFuncVar->arg_begin();
          for (unsigned i = 0, e = DSAMerge.size(); i != e; ++i, ++AI)
            AI->setName(DSAMerge[i]->getName());

          // Rewrite branches to basic blocks outside of the loop to new dummy blocks
          // within the new function. This must be done before we lose track of which
          // blocks were originally in the code region.
          // ?? FIXME: Parece que esto se usa para cambiar los branches al codigo que movemos
          //           Por ej. br label %codeRepl
          std::vector<User *> Users(header->user_begin(), header->user_end());
          for (unsigned i = 0, e = Users.size(); i != e; ++i)
            // The BasicBlock which contains the branch is not in the region
            // modify the branch target to a new block
            if (Instruction *I = dyn_cast<Instruction>(Users[i]))
              if (I->isTerminator() && !Blocks.count(I->getParent()) &&
                  I->getParent()->getParent() == oldFunction)
                I->replaceUsesOfWith(header, newHeader);

          // nanos6_ol_task_region_*
          paramTy.clear();
          paramTy.push_back(TaskArgsTy->getPointerTo());
          paramTy.push_back(Type::getInt8PtrTy(M->getContext())); /* void * device_env */
          paramTy.push_back(TaskAddrTranslationEntryTy->getPointerTo()); /* nanos6_address_translation_entry_t *address_translation_table */
          FunctionType *outlineFuncType =
                          FunctionType::get(RetTy, paramTy, /*IsVarArgs */ false);

          Function *outlineFuncVar = Function::Create(
              outlineFuncType, GlobalValue::InternalLinkage, oldFunction->getAddressSpace(),
              "nanos6_ol_task_region_" + oldFunction->getName(), M);
          BasicBlock *outlineEntryBB = BasicBlock::Create(M->getContext(), "entry", outlineFuncVar);

          IRBuilder<> BBBuilder(outlineEntryBB);
          // BBBuilder.SetInsertPoint(outlineEntryBB);

          // First arg is the nanos_task_args
          AI = outlineFuncVar->arg_begin();
          Value *outlineFuncTaskArgs = &*AI++;
          SmallVector<Value *, 4> TaskUnpackParams;
          size_t TaskArgsIdx = 0;;
          for (unsigned i = 0; i < TI.DSAInfo.Shared.size(); ++i, ++TaskArgsIdx) {
            Value *Idx[2];
            Idx[0] = Constant::getNullValue(Type::getInt32Ty(M->getContext()));
            Idx[1] = ConstantInt::get(Type::getInt32Ty(M->getContext()), TaskArgsIdx);
            Value *GEP = BBBuilder.CreateGEP(
                outlineFuncTaskArgs, Idx, "gep_" + TI.DSAInfo.Shared[i]->getName());
            Value *LGEP = BBBuilder.CreateLoad(GEP, "load_" + GEP->getName());
            TaskUnpackParams.push_back(LGEP);
          }
          for (unsigned i = 0; i < TI.DSAInfo.Private.size(); ++i, ++TaskArgsIdx) {
            Value *Idx[2];
            Idx[0] = Constant::getNullValue(Type::getInt32Ty(M->getContext()));
            Idx[1] = ConstantInt::get(Type::getInt32Ty(M->getContext()), TaskArgsIdx);
            Value *GEP = BBBuilder.CreateGEP(
                outlineFuncTaskArgs, Idx, "gep_" + TI.DSAInfo.Private[i]->getName());
            TaskUnpackParams.push_back(GEP);
          }
          for (unsigned i = 0; i < TI.DSAInfo.Firstprivate.size(); ++i, ++TaskArgsIdx) {
            Value *Idx[2];
            Idx[0] = Constant::getNullValue(Type::getInt32Ty(M->getContext()));
            Idx[1] = ConstantInt::get(Type::getInt32Ty(M->getContext()), TaskArgsIdx);
            Value *GEP = BBBuilder.CreateGEP(
                outlineFuncTaskArgs, Idx, "gep_" + TI.DSAInfo.Firstprivate[i]->getName());
            TaskUnpackParams.push_back(GEP);
          }
          TaskUnpackParams.push_back(&*AI++);
          TaskUnpackParams.push_back(&*AI++);
          Instruction *TaskUnpackCall =
            BBBuilder.CreateCall(unpackFuncVar, TaskUnpackParams);
          ReturnInst *TaskOlRet = BBBuilder.CreateRetVoid();

          // 0.1 Create Nanos6 task data structures info
          TaskInvInfoVar = M->getOrInsertGlobal(("task_invocation_info_" + oldFunction->getName()).str(),
                                            TaskInvInfoTy,
                                            [&] {
            GlobalVariable *GV = new GlobalVariable(*M, TaskInvInfoTy,
                                      false,
                                      GlobalVariable::InternalLinkage,
                                      ConstantStruct::get(TaskInvInfoTy,
                                                          ConstantPointerNull::get(Type::getInt8PtrTy(M->getContext()))),
                                      ("task_invocation_info_" + oldFunction->getName()).str());
            GV->setAlignment(64);
            return GV;
          });

          TaskImplInfoVar = M->getOrInsertGlobal(("implementations_var_" + oldFunction->getName()).str(),
                                            ArrayType::get(TaskImplInfoTy, 1),
                                            [&] {
            auto *outlineFuncCastTy = FunctionType::get(Type::getVoidTy(M->getContext()),
                                                    {Type::getInt8PtrTy(M->getContext()), /* void * */
                                                     Type::getInt8PtrTy(M->getContext()), /* void * */
                                                     TaskAddrTranslationEntryTy->getPointerTo()
                                                     }, false);
            GlobalVariable *GV = new GlobalVariable(*M, ArrayType::get(TaskImplInfoTy, 1),
                                      false,
                                      GlobalVariable::InternalLinkage,
                                      ConstantArray::get(ArrayType::get(TaskImplInfoTy, 1),
                                                         ConstantStruct::get(TaskImplInfoTy,
                                                                             ConstantInt::get(Type::getInt32Ty(M->getContext()), 0),
                                                                             ConstantExpr::getPointerCast(outlineFuncVar, outlineFuncCastTy->getPointerTo()),
                                                                             ConstantPointerNull::get(GetConstraintsFuncTy->getPointerTo()),
                                                                             ConstantPointerNull::get(Type::getInt8PtrTy(M->getContext())),
                                                                             ConstantPointerNull::get(Type::getInt8PtrTy(M->getContext())),
                                                                             ConstantPointerNull::get(RunWrapperFuncTy->getPointerTo()))),
                                      ("implementations_var_" + oldFunction->getName()).str());

            GV->setAlignment(64);
            return GV;
          });
          TaskInfoVar = M->getOrInsertGlobal(("task_info_var_" + oldFunction->getName()).str(),
                                            TaskInfoTy,
                                            [&] {
            GlobalVariable *GV = new GlobalVariable(*M, TaskInfoTy,
                                      false,
                                      GlobalVariable::InternalLinkage,
                                      ConstantStruct::get(TaskInfoTy,
                                                          ConstantInt::get(Type::getInt32Ty(M->getContext()), -1),
                                                          ConstantPointerNull::get(RegisterInfoFuncTy->getPointerTo()),
                                                          ConstantPointerNull::get(GetPriorityFuncTy->getPointerTo()),
                                                          ConstantPointerNull::get(Type::getInt8PtrTy(M->getContext())),
                                                          ConstantInt::get(Type::getInt32Ty(M->getContext()), 1),
                                                          ConstantExpr::getPointerCast(TaskImplInfoVar, TaskImplInfoTy->getPointerTo()),
                                                          ConstantPointerNull::get(DestroyArgsBlockFuncTy->getPointerTo()),
                                                          ConstantPointerNull::get(DuplicateArgsBlockFuncTy->getPointerTo()),
                                                          ConstantPointerNull::get(ReductInitsFuncTy->getPointerTo()),
                                                          ConstantPointerNull::get(ReductCombsFuncTy->getPointerTo())),
                                      ("task_info_var_" + oldFunction->getName()).str());

            GV->setAlignment(64);
            return GV;
          });

          return unpackFuncVar;
        };
        auto emitCaptureAndCall = [&](Function *newFunction,
                                      BasicBlock *codeReplacer,
                                      const SetVector<BasicBlock *> &Blocks) {

          IRBuilder<> IRB(codeReplacer);
          Function *CreateTaskFuncTy = cast<Function>(M.getOrInsertFunction("nanos6_create_task",
              IRB.getVoidTy(),
              TaskInfoTy->getPointerTo(),
              TaskInvInfoTy->getPointerTo(),
              IRB.getInt64Ty(), /* size_t args_lock_size */
              IRB.getInt8PtrTy()->getPointerTo(),
              IRB.getInt8PtrTy()->getPointerTo(),
              IRB.getInt64Ty(), /* size_t flags */
              IRB.getInt64Ty())); /* size_t num_deps */
          Value *TaskArgsVar = IRB.CreateAlloca(TaskArgsTy->getPointerTo());
          Value *TaskArgsVarCast = IRB.CreateBitCast(TaskArgsVar, IRB.getInt8PtrTy()->getPointerTo());
          // Value *TaskFlagsVar = IRB.CreateAlloca(IRB.getInt64Ty());
          // IRB.CreateStore(ConstantInt::get(IRB.getInt64Ty(), 0), TaskFlagsVar);
          Value *TaskPtrVar = IRB.CreateAlloca(IRB.getInt8PtrTy());
          // Value *TaskNumDepsVar = IRB.CreateAlloca(IRB.getInt64Ty());
          // IRB.CreateStore(ConstantInt::get(IRB.getInt64Ty(), 0), TaskNumDepsVar);
          uint64_t TaskArgsSizeOf = M.getDataLayout().getTypeAllocSize(TaskArgsTy);
          IRB.CreateCall(CreateTaskFuncTy, {TaskInfoVar,
                                      TaskInvInfoVar,
                                      ConstantInt::get(IRB.getInt64Ty(), TaskArgsSizeOf),
                                      TaskArgsVarCast,
                                      TaskPtrVar,
                                      ConstantInt::get(IRB.getInt64Ty(), 0), // TaskFlagsVar,
                                      ConstantInt::get(IRB.getInt64Ty(), 0)}); // TaskNumDepsVar});

          // DSA capture
          Value *TaskArgsVarL = IRB.CreateLoad(TaskArgsVar);
          size_t TaskArgsIdx = 0;
          for (unsigned i = 0; i < TI.DSAInfo.Shared.size(); ++i, ++TaskArgsIdx) {
            Value *Idx[2];
            Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
            Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsIdx);
            Value *GEP = IRB.CreateGEP(
                TaskArgsVarL, Idx, "gep_" + TI.DSAInfo.Shared[i]->getName());
            Value *CaptureDSA = IRB.CreateStore(TI.DSAInfo.Shared[i], GEP);
          }
          TaskArgsIdx += TI.DSAInfo.Private.size();
          for (unsigned i = 0; i < TI.DSAInfo.Firstprivate.size(); ++i, ++TaskArgsIdx) {
            Value *Idx[2];
            Idx[0] = Constant::getNullValue(IRB.getInt32Ty());
            Idx[1] = ConstantInt::get(IRB.getInt32Ty(), TaskArgsIdx);
            Value *GEP = IRB.CreateGEP(
                TaskArgsVarL, Idx, "gep_" + TI.DSAInfo.Firstprivate[i]->getName());
            Value *FPValue = IRB.CreateLoad(TI.DSAInfo.Firstprivate[i]);
            Value *CaptureDSA = IRB.CreateStore(FPValue, GEP);
          }

          Function *TaskSubmitFuncTy = cast<Function>(M.getOrInsertFunction("nanos6_submit_task",
              IRB.getVoidTy(),
              IRB.getInt8PtrTy()));
          Value *TaskPtrVarL = IRB.CreateLoad(TaskPtrVar);
          IRB.CreateCall(TaskSubmitFuncTy, TaskPtrVarL);

          // Since there may be multiple exits from the original region, make the new
          // function return an unsigned, switch on that number.  This loop iterates
          // over all of the blocks in the extracted region, updating any terminator
          // instructions in the to-be-extracted region that branch to blocks that are
          // not in the region to be extracted.
          std::map<BasicBlock *, BasicBlock *> ExitBlockMap;

          unsigned switchVal = 0;
          for (BasicBlock *Block : Blocks) {
            Instruction *TI = Block->getTerminator();
            for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
              if (!Blocks.count(TI->getSuccessor(i))) {
                BasicBlock *OldTarget = TI->getSuccessor(i);

                IRB.CreateBr(OldTarget);

                // add a new basic block which returns the appropriate value
                BasicBlock *&NewTarget = ExitBlockMap[OldTarget];
                if (!NewTarget) {
                  // If we don't already have an exit stub for this non-extracted
                  // destination, create one now!
                  NewTarget = BasicBlock::Create(M.getContext(),
                                                 OldTarget->getName() + ".exitStub",
                                                 newFunction);

                  ReturnInst::Create(M.getContext(), nullptr, NewTarget);
                }
                // rewrite the original branch instruction with this new target
                TI->setSuccessor(i, NewTarget);
              }
          }

          return nullptr;
        };
        CodeExtractor CE(TaskBBs, constructOmpSsFunctions, emitCaptureAndCall);

        // SetVector<Value *> DSAMerge;
        // DSAMerge.insert(TI.DSAInfo.Shared.begin(), TI.DSAInfo.Shared.end());
        // DSAMerge.insert(TI.DSAInfo.Private.begin(), TI.DSAInfo.Private.end());
        // DSAMerge.insert(TI.DSAInfo.Firstprivate.begin(), TI.DSAInfo.Firstprivate.end());
        // CE.SetInputs(DSAMerge, {}, {}, {});
        #endif

        Function *OutF = CE.extractCodeRegion();
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
