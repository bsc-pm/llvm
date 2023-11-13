#include "llvm/Transforms/OmpSs/Nanos6API.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

namespace llvm {
namespace nanos6Api {

Nanos6LoopBounds& Nanos6LoopBounds::getInstance(Module &M) {
  static auto instance = std::unique_ptr<Nanos6LoopBounds>(nullptr);
  if (!instance) {
    instance.reset(new Nanos6LoopBounds);
    instance->Ty = StructType::create(M.getContext(),
      "nanos6_loop_bounds_t");

    // size_t lower_bound;
    // size_t upper_bound;
    // size_t grainsize;
    // size_t chunksize;

    instance->LBoundTy = Type::getInt64Ty(M.getContext());
    instance->UBoundTy = Type::getInt64Ty(M.getContext());
    instance->GrainsizeTy = Type::getInt64Ty(M.getContext());
    instance->ChunksizeTy = Type::getInt64Ty(M.getContext());

    instance->Ty->setBody({
      instance->LBoundTy,
      instance->UBoundTy,
      instance->GrainsizeTy,
      instance->ChunksizeTy
    });
  }
  return *instance.get();
}

Nanos6TaskAddrTranslationEntry& Nanos6TaskAddrTranslationEntry::getInstance(Module &M) {
  static auto instance = std::unique_ptr<Nanos6TaskAddrTranslationEntry>(nullptr);
  if (!instance) {
    instance.reset(new Nanos6TaskAddrTranslationEntry);
    instance->Ty = StructType::create(M.getContext(),
      "nanos6_address_translation_entry_t");

    // size_t local_address
    // size_t device_address
    instance->LocalAddrTy = Type::getInt64Ty(M.getContext());
    instance->DeviceAddrTy = Type::getInt64Ty(M.getContext());

    instance->Ty->setBody({
      instance->LocalAddrTy,
      instance->DeviceAddrTy
    });
  }
  return *instance.get();
}

Nanos6TaskConstraints& Nanos6TaskConstraints::getInstance(Module &M) {
  static auto instance = std::unique_ptr<Nanos6TaskConstraints>(nullptr);
  if (!instance) {
    instance.reset(new Nanos6TaskConstraints);
    instance->Ty = StructType::create(M.getContext(),
      "nanos6_task_constraints_t");

    // size_t cost
    instance->CostTy = Type::getInt64Ty(M.getContext());

    instance->Ty->setBody(instance->CostTy);
  }
  return *instance.get();
}

Nanos6TaskInvInfo& Nanos6TaskInvInfo::getInstance(Module &M) {
  static auto instance = std::unique_ptr<Nanos6TaskInvInfo>(nullptr);
  if (!instance) {
    instance.reset(new Nanos6TaskInvInfo);
    instance->Ty = StructType::create(M.getContext(),
      "nanos6_task_invocation_info_t");

    // const char *invocation_source
    instance->InvSourceTy = PointerType::getUnqual(M.getContext());

    instance->Ty->setBody(instance->InvSourceTy);
  }
  return *instance.get();
}

Nanos6TaskImplInfo& Nanos6TaskImplInfo:: getInstance(Module &M) {
  static auto instance = std::unique_ptr<Nanos6TaskImplInfo>(nullptr);
  if (!instance) {
    instance.reset(new Nanos6TaskImplInfo);
    instance->Ty = StructType::create(M.getContext(),
      "nanos6_task_implementation_info_t");

    // int device_type_id;
    instance->DeviceTypeIdTy = Type::getInt32Ty(M.getContext());
    // void (*run)(void *, void *, nanos6_address_translation_entry_t *);
    instance->RunFuncTy = PointerType::getUnqual(M.getContext());
    // void (*get_constraints)(void *, nanos6_task_constraints_t *);
    instance->GetConstraintsFuncTy = PointerType::getUnqual(M.getContext());
    // const char *task_label;
    instance->TaskLabelTy = PointerType::getUnqual(M.getContext());
    // const char *declaration_source;
    instance->DeclSourceTy = PointerType::getUnqual(M.getContext());
    // const char *device_function_name;
    instance->DevFuncTy = PointerType::getUnqual(M.getContext());
    instance->Ty->setBody({
      instance->DeviceTypeIdTy, instance->RunFuncTy,
      instance->GetConstraintsFuncTy, instance->TaskLabelTy,
      instance->DeclSourceTy, instance->DevFuncTy
    });
  }
  return *instance.get();
}

Nanos6TaskInfo& Nanos6TaskInfo::getInstance(Module &M) {
  static auto instance = std::unique_ptr<Nanos6TaskInfo>(nullptr);
  if (!instance) {
    instance.reset(new Nanos6TaskInfo);
    instance->Ty = StructType::create(M.getContext(),
      "nanos6_task_info_t");

    // int num_symbols;
    instance->NumSymbolsTy = Type::getInt32Ty(M.getContext());;
    // void (*register_depinfo)(void *, void *);
    instance->RegisterInfoFuncTy = PointerType::getUnqual(M.getContext());
    // void (*onready_action)(void *args_block);
    instance->OnreadyActionFuncTy = PointerType::getUnqual(M.getContext());
    // void (*get_priority)(void *, nanos6_priority_t *);
    // void (*get_priority)(void *, long int *);
    instance->GetPriorityFuncTy = PointerType::getUnqual(M.getContext());
    // int implementation_count;
    instance->ImplCountTy = Type::getInt32Ty(M.getContext());
    // nanos6_task_implementation_info_t *implementations;
    instance->TaskImplInfoTy = PointerType::getUnqual(M.getContext());
    // void (*destroy_args_block)(void *);
    instance->DestroyArgsBlockFuncTy = PointerType::getUnqual(M.getContext());
    // void (*duplicate_args_block)(const void *, void **);
    instance->DuplicateArgsBlockFuncTy = PointerType::getUnqual(M.getContext());
    // void (**reduction_initializers)(void *, void *, size_t);
    instance->ReductInitsFuncTy = PointerType::getUnqual(M.getContext());
    // void (**reduction_combiners)(void *, void *, size_t);
    instance->ReductCombsFuncTy = PointerType::getUnqual(M.getContext());
    // void *task_type_data;
    instance->TaskTypeDataTy = PointerType::getUnqual(M.getContext());
    // void (*iter_condition)(void *, uint8_t *);
    instance->IterConditionFuncTy = PointerType::getUnqual(M.getContext());
    // int num_args;
    instance->NumArgsTy = Type::getInt32Ty(M.getContext());
    // int *sizeof_table;
    instance->SizeofTableDataTy = PointerType::getUnqual(M.getContext());
    // int *offset_table;
    instance->OffsetTableDataTy = PointerType::getUnqual(M.getContext());
    // int *arg_idx_table;
    instance->ArgIdxTableDataTy = PointerType::getUnqual(M.getContext());

    instance->Ty->setBody({
      instance->NumSymbolsTy, instance->RegisterInfoFuncTy, instance->OnreadyActionFuncTy, instance->GetPriorityFuncTy,
      instance->ImplCountTy,
      instance->TaskImplInfoTy,
      instance->DestroyArgsBlockFuncTy,
      instance->DuplicateArgsBlockFuncTy,
      instance->ReductInitsFuncTy,
      instance->ReductCombsFuncTy,
      instance->TaskTypeDataTy,
      instance->IterConditionFuncTy,
      instance->NumArgsTy,
      instance->SizeofTableDataTy,
      instance->OffsetTableDataTy,
      instance->ArgIdxTableDataTy
    });
  }
  return *instance.get();
}

Nanos6Version& Nanos6Version::getInstance(Module &M) {
  static auto instance = std::unique_ptr<Nanos6Version>(nullptr);
  if (!instance) {
    instance.reset(new Nanos6Version);
    instance->Ty = StructType::create(M.getContext(),
      "nanos6_version_t");

    // uint64_t family;
    instance->FamilyTy = Type::getInt64Ty(M.getContext());
    // uint64_t version;
    instance->VersionTy = Type::getInt64Ty(M.getContext());

    instance->Ty->setBody({
      instance->FamilyTy, instance->VersionTy
    });
  }
  return *instance.get();
}

StringRef Nanos6MultidepFactory::getDependTypeStrFromType(
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

FunctionType *Nanos6MultidepFactory::BuildDepFuncType(
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

FunctionType *Nanos6MultidepFactory::BuildReleaseDepFuncType(
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

FunctionCallee Nanos6MultidepFactory::getMultidepFuncCallee(
    Module &M, DependInfo::DependType DType, size_t Ndims, bool IsReduction) {
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

FunctionCallee Nanos6MultidepFactory::getReleaseMultidepFuncCallee(
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

FunctionCallee createTaskFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_create_task",
    Type::getVoidTy(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    Type::getInt64Ty(M.getContext())
  );
}

FunctionCallee taskSubmitFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_submit_task",
    Type::getVoidTy(M.getContext()),
    PointerType::getUnqual(M.getContext())
  );
}

FunctionCallee createLoopFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_create_loop",
    Type::getVoidTy(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    Type::getInt64Ty(M.getContext())
  );
}

FunctionCallee createIterFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_create_iter",
    Type::getVoidTy(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    Type::getInt64Ty(M.getContext())
  );
}

FunctionCallee taskInFinalFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_in_final",
    Type::getInt32Ty(M.getContext())
  );
}

FunctionCallee taskInfoRegisterFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_register_task_info",
    Type::getVoidTy(M.getContext()),
    PointerType::getUnqual(M.getContext())
  );
}

FunctionCallee userLockFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_user_lock",
    Type::getVoidTy(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    PointerType::getUnqual(M.getContext())
  );
}

FunctionCallee userUnlockFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_user_unlock",
    Type::getVoidTy(M.getContext()),
    PointerType::getUnqual(M.getContext())
  );
}

FunctionCallee taskInfoRegisterCtorFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_constructor_register_task_info",
    Type::getVoidTy(M.getContext())
  );
}

FunctionCallee registerCtorAssertFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_constructor_register_assert",
    Type::getVoidTy(M.getContext())
  );
}

FunctionCallee registerAssertFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_config_assert",
    Type::getVoidTy(M.getContext()),
    PointerType::getUnqual(M.getContext())
  );
}

FunctionCallee registerCtorCheckVersionFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_constructor_check_version",
    Type::getVoidTy(M.getContext())
  );
}

FunctionCallee checkVersionFuncCallee(Module &M) {
  return M.getOrInsertFunction("nanos6_check_version",
    Type::getVoidTy(M.getContext()),
    Type::getInt64Ty(M.getContext()),
    PointerType::getUnqual(M.getContext()),
    PointerType::getUnqual(M.getContext())
  );
}

} // end namospace nanos6Api
} // end namospace llvm
