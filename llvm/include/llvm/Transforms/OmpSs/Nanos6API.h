#ifndef LLVM_TRANSFORMS_NANOS6API_H
#define LLVM_TRANSFORMS_NANOS6API_H

#include "llvm/Analysis/OmpSsRegionAnalysis.h"

namespace llvm {
class FunctionCallee;
class Module;
class Type;
class StructType;
namespace nanos6Api {

class Nanos6LoopBounds {
private:
  StructType *Ty;
  Type *LBoundTy;
  Type *UBoundTy;
  Type *GrainsizeTy;
  Type *ChunksizeTy;

  Nanos6LoopBounds(){};
  Nanos6LoopBounds(const Nanos6LoopBounds&){};
public:
  ~Nanos6LoopBounds(){};

  static Nanos6LoopBounds& getInstance(Module &M);
  StructType *getType() const { return Ty; }
  Type *getLBType() const { return LBoundTy; }
  Type *getUBType() const { return UBoundTy; }
  Type *getGrainsizeType() const { return GrainsizeTy; }
  Type *getChunksizeType() const { return ChunksizeTy; }
};

class Nanos6TaskAddrTranslationEntry {
private:
  StructType *Ty;
  Type *LocalAddrTy;
  Type *DeviceAddrTy;

  Nanos6TaskAddrTranslationEntry(){};
  Nanos6TaskAddrTranslationEntry(const Nanos6TaskAddrTranslationEntry&){};
public:
  ~Nanos6TaskAddrTranslationEntry(){};

  static Nanos6TaskAddrTranslationEntry& getInstance(Module &M);
  StructType *getType() const { return Ty; }
  Type *getLocalAddrType() const { return LocalAddrTy; }
  Type *getDeviceAddrType() const { return DeviceAddrTy; }
};

class Nanos6TaskConstraints {
private:
  StructType *Ty;
  Type *CostTy;

  Nanos6TaskConstraints(){};
  Nanos6TaskConstraints(const Nanos6TaskConstraints&){};
public:
  ~Nanos6TaskConstraints(){};

  static Nanos6TaskConstraints& getInstance(Module &M);
  StructType *getType() const { return Ty; }
  Type *getCostType() const { return CostTy; }
};

class Nanos6TaskInvInfo {
private:
  StructType *Ty;
  Type *InvSourceTy;

  Nanos6TaskInvInfo(){};
  Nanos6TaskInvInfo(const Nanos6TaskInvInfo&){};
public:
  ~Nanos6TaskInvInfo(){};

  static Nanos6TaskInvInfo& getInstance(Module &M);
  StructType *getType() const { return Ty; }
  Type *getInvSourceType() const { return InvSourceTy; }
};

class Nanos6TaskImplInfo {
private:
  StructType *Ty;
  Type *DeviceTypeIdTy;
  Type *RunFuncTy;
  Type *GetConstraintsFuncTy;
  Type *TaskLabelTy;
  Type *DeclSourceTy;
  Type *DevFuncTy;

  Nanos6TaskImplInfo(){};
  Nanos6TaskImplInfo(const Nanos6TaskImplInfo&){};
public:
  ~Nanos6TaskImplInfo(){};

  static Nanos6TaskImplInfo& getInstance(Module &M);
  StructType *getType() const { return Ty; }
  Type *getDeviceTypeIdType() const { return DeviceTypeIdTy; }
  Type *getRunFuncType() const { return RunFuncTy; }
  Type *getGetConstraintsFuncType() const { return GetConstraintsFuncTy; }
  Type *getTaskLabelType() const { return TaskLabelTy; }
  Type *getDeclSourceType() const { return DeclSourceTy; }
  Type *getDevFuncType() const { return DevFuncTy; }
};

class Nanos6TaskInfo {
private:
  StructType *Ty;
  Type *NumSymbolsTy;
  Type *RegisterInfoFuncTy;
  Type *OnreadyActionFuncTy;
  Type *GetPriorityFuncTy;
  Type *ImplCountTy;
  Type *TaskImplInfoTy;
  Type *DestroyArgsBlockFuncTy;
  Type *DuplicateArgsBlockFuncTy;
  Type *ReductInitsFuncTy;
  Type *ReductCombsFuncTy;
  Type *TaskTypeDataTy;
  Type *IterConditionFuncTy;
  Type *NumArgsTy;
  Type *SizeofTableDataTy;
  Type *OffsetTableDataTy;
  Type *ArgIdxTableDataTy;
  Type *CoroHandleIdxDataTy;

  Nanos6TaskInfo(){};
  Nanos6TaskInfo(const Nanos6TaskInfo&){};
public:
  ~Nanos6TaskInfo(){};

  static Nanos6TaskInfo& getInstance(Module &M);
  StructType *getType() const { return Ty; }
  Type *getNumSymbolsType() const { return NumSymbolsTy; }
  Type *getRegisterInfoFuncType() const { return RegisterInfoFuncTy; }
  Type *getOnreadyActionFuncType() const { return OnreadyActionFuncTy; }
  Type *getGetPriorityFuncType() const { return GetPriorityFuncTy; }
  Type *getImplCountType() const { return ImplCountTy; }
  Type *getTaskImplInfoType() const { return TaskImplInfoTy; }
  Type *getDestroyArgsBlockFuncType() const { return DestroyArgsBlockFuncTy; }
  Type *getDuplicateArgsBlockFuncType() const { return DuplicateArgsBlockFuncTy; }
  Type *getReductInitsFuncType() const { return ReductInitsFuncTy; }
  Type *getReductCombsFuncType() const { return ReductCombsFuncTy; }
  Type *getTaskTypeDataType() const { return TaskTypeDataTy; }
  Type *getIterConditionFuncType() const { return IterConditionFuncTy; }
  Type *getNumArgsType() const { return NumArgsTy; }
  Type *getSizeofTableDataType() const { return SizeofTableDataTy; }
  Type *getOffsetTableDataType() const { return OffsetTableDataTy; }
  Type *getArgIdxTableDataType() const { return ArgIdxTableDataTy; }
  Type *getCoroHandleIdxDataType() const { return CoroHandleIdxDataTy; }
};

class Nanos6Version {
private:
  StructType *Ty;
  Type *FamilyTy;
  Type *VersionTy;

  Nanos6Version(){};
  Nanos6Version(const Nanos6TaskInfo&){};
public:
  ~Nanos6Version(){};

  static Nanos6Version& getInstance(Module &M);
  StructType *getType() const { return Ty; }
  Type *getFamilyType() const { return FamilyTy; }
  Type *getVersionType() const { return VersionTy; }
};

class Nanos6MultidepFactory {
  const size_t MAX_DEP_DIMS = 8;
private:
  StringMap<FunctionCallee> DepNameToFuncCalleeMap;

  static StringRef getDependTypeStrFromType(DependInfo::DependType DType);
  FunctionType *BuildDepFuncType(
    Module &M, StringRef FullName, size_t Ndims, bool IsReduction);
  FunctionType *BuildReleaseDepFuncType(
    Module &M, StringRef FullName, size_t Ndims);
public:
  FunctionCallee getMultidepFuncCallee(
      Module &M, DependInfo::DependType DType, size_t Ndims, bool IsReduction=false);
  FunctionCallee getReleaseMultidepFuncCallee(
      Module &M, DependInfo::DependType DType, size_t Ndims);
};

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
FunctionCallee createTaskFuncCallee(Module &M);

// void nanos6_submit_task(void *task);
FunctionCallee taskSubmitFuncCallee(Module &M);

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
FunctionCallee createLoopFuncCallee(Module &M);

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
FunctionCallee createIterFuncCallee(Module &M);

// int nanos6_in_final(void);
FunctionCallee taskInFinalFuncCallee(Module &M);

// void nanos6_register_task_info(nanos6_task_info_t *task_info);
FunctionCallee taskInfoRegisterFuncCallee(Module &M);

// void nanos6_user_lock(void **handlerPointer, const char *invocation_source);
FunctionCallee userLockFuncCallee(Module &M);

// void nanos6_user_unlock(void **handlerPointer);
FunctionCallee userUnlockFuncCallee(Module &M);

// void nanos6_constructor_register_task_info(void);
// NOTE: This does not belong to nanos6 API
FunctionCallee taskInfoRegisterCtorFuncCallee(Module &M);

// void nanos6_constructor_register_assert(void);
// NOTE: This does not belong to nanos6 API
FunctionCallee registerCtorAssertFuncCallee(Module &M);

// void nanos6_config_assert(const char *str);
FunctionCallee registerAssertFuncCallee(Module &M);

// void nanos6_constructor_check_version(void);
// NOTE: This does not belong to nanos6 API
FunctionCallee registerCtorCheckVersionFuncCallee(Module &M);

// void nanos6_check_version(uint64_t size, nanos6_version_t *arr, const char *source);
FunctionCallee checkVersionFuncCallee(Module &M);

// void nanos6_suspend();
FunctionCallee suspendFuncCallee(Module &M);

}
}

#endif // LLVM_TRANSFORMS_NANOS6API_H

