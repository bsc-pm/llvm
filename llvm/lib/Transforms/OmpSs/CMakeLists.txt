add_llvm_library(LLVMOmpSs
  OmpSsTransform.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/OmpSs

  DEPENDS
  intrinsics_gen
  )
