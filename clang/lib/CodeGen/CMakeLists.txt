set(LLVM_LINK_COMPONENTS
  Analysis
  BitReader
  BitWriter
  Core
  Coroutines
  Coverage
  FrontendOpenMP
  IPO
  IRReader
  AggressiveInstCombine
  InstCombine
  Instrumentation
  LTO
  Linker
  MC
  ObjCARCOpts
  Object
  Passes
  ProfileData
  Remarks
  ScalarOpts
  Support
  Target
  TransformUtils
  )

# In a standard Clang+LLVM build, we need to generate intrinsics before
# building codegen. In a standalone build, LLVM is already built and we don't
# need this dependency. Furthermore, LLVM doesn't export it so we can't have
# this dependency.
set(codegen_deps intrinsics_gen)
if (CLANG_BUILT_STANDALONE)
  set(codegen_deps)
endif()

if (MSVC)
  set_source_files_properties(CodeGenModule.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()

add_clang_library(clangCodeGen
  BackendUtil.cpp
  CGAtomic.cpp
  CGBlocks.cpp
  CGBuiltin.cpp
  CGCUDANV.cpp
  CGCUDARuntime.cpp
  CGCXX.cpp
  CGCXXABI.cpp
  CGCall.cpp
  CGClass.cpp
  CGCleanup.cpp
  CGCoroutine.cpp
  CGDebugInfo.cpp
  CGDecl.cpp
  CGDeclCXX.cpp
  CGException.cpp
  CGExpr.cpp
  CGExprAgg.cpp
  CGExprCXX.cpp
  CGExprComplex.cpp
  CGExprConstant.cpp
  CGExprScalar.cpp
  CGGPUBuiltin.cpp
  CGLoopInfo.cpp
  CGNonTrivialStruct.cpp
  CGObjC.cpp
  CGObjCGNU.cpp
  CGObjCMac.cpp
  CGObjCRuntime.cpp
  CGOpenCLRuntime.cpp
  CGOpenMPRuntime.cpp
  CGOpenMPRuntimeNVPTX.cpp
  CGRecordLayoutBuilder.cpp
  CGStmt.cpp
  CGStmtOpenMP.cpp
  CGVTT.cpp
  CGVTables.cpp
  CodeGenABITypes.cpp
  CodeGenAction.cpp
  CodeGenFunction.cpp
  CodeGenModule.cpp
  CodeGenPGO.cpp
  CodeGenTBAA.cpp
  CodeGenTypes.cpp
  ConstantInitBuilder.cpp
  CoverageMappingGen.cpp
  ItaniumCXXABI.cpp
  MacroPPCallbacks.cpp
  MicrosoftCXXABI.cpp
  ModuleBuilder.cpp
  ObjectFilePCHContainerOperations.cpp
  PatternInit.cpp
  SanitizerMetadata.cpp
  SwiftCallingConv.cpp
  TargetInfo.cpp
  VarBypassDetector.cpp

  ENABLE_PLUGINS

  DEPENDS
  ${codegen_deps}
  builtins_codegen_epi

  LINK_LIBS
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  )

# EPI
# FIXME: Do this better
if( ${CLANG_TABLEGEN} STREQUAL "clang-tblgen" )
  set(REAL_CLANG_TABLEGEN $<TARGET_FILE:clang-tblgen>)
else()
  set(REAL_CLANG_TABLEGEN ${CLANG_TABLEGEN_EXE})
endif()

add_custom_command(
  OUTPUT EPIBuiltinCodegen.inc
  COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/EPI/generate-epi-builtins-def.py
    --mode codegen
    --tablegen ${REAL_CLANG_TABLEGEN}
    --output-file EPIBuiltinCodegen.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/clang/Basic/epi_builtins.td
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/clang/Basic/epi_builtins.td
     ${CLANG_TABLEGEN_EXE}
     ${CLANG_TABLEGEN_TARGET}
     ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/EPI/generate-epi-builtins-def.py
   VERBATIM)
add_custom_target(builtins_codegen_epi ALL DEPENDS EPIBuiltinCodegen.inc)
