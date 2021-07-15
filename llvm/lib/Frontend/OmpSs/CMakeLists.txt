add_llvm_component_library(LLVMFrontendOmpSs
  OSS.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/OmpSs

  DEPENDS
  intrinsics_gen
  oss_gen

  LINK_COMPONENTS
  Core
  Support
  TransformUtils
  )
