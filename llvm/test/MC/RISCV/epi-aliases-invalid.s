# RUN: not llvm-mc --triple=riscv64 -mattr +v < %s --show-encoding 2>&1 \
# RUN:    | FileCheck --check-prefix=ALIAS %s
# RUN: not llvm-mc --triple=riscv64 -mattr=+v --riscv-no-aliases < %s \
# RUN:    --show-encoding 2>&1 | FileCheck --check-prefix=NO-ALIAS %s

# Note: In this test we check the error messages given when exceeding the
# immediate ranges for the alias.

# ALIAS:    error: immediate must be in the range [-15, 16]
# NO-ALIAS: error: immediate must be in the range [-15, 16]
vmslt.vi v1, v2, -16
# ALIAS:    error: immediate must be in the range [-15, 16]
# NO-ALIAS: error: immediate must be in the range [-15, 16]
vmslt.vi v1, v2, 17
