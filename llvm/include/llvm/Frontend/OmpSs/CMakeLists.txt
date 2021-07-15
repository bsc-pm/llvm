set(LLVM_TARGET_DEFINITIONS OSS.td)
tablegen(LLVM OSS.h.inc --gen-directive-decl)
tablegen(LLVM OSS.inc --gen-directive-impl)
add_public_tablegen_target(oss_gen)
