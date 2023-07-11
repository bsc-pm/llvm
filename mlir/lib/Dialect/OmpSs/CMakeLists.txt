add_mlir_dialect_library(MLIROmpSs
  IR/OmpSsDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OmpSs

  DEPENDS
  MLIROmpSsOpsIncGen
  MLIROmpSsOpsInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
