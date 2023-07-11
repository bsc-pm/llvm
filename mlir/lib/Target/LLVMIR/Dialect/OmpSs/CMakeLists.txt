add_mlir_translation_library(MLIROmpSsToLLVMIRTranslation
  OmpSsToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIROmpSs
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
