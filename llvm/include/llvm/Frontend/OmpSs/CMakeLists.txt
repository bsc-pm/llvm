set(LLVM_TARGET_DEFINITIONS OSS.td)
tablegen(LLVM OSS.h.inc --gen-directive-decl)
tablegen(LLVM OSS.cpp.inc --gen-directive-gen)
add_public_tablegen_target(oss_gen)
