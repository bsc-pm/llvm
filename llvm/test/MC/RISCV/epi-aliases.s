# RUN: llvm-mc --triple=riscv64 -mattr +v < %s --show-encoding 2>&1 \
# RUN:    | FileCheck --check-prefix=ALIAS %s
# RUN: llvm-mc --triple=riscv64 -mattr=+v --riscv-no-aliases < %s \
# RUN:    --show-encoding 2>&1 | FileCheck --check-prefix=NO-ALIAS %s

# ALIAS:    vsetvli s0, t3, e32,m1          # encoding: [0x57,0x74,0x8e,0x00]
# NO-ALIAS: vsetvli s0, t3, e32,m1          # encoding: [0x57,0x74,0x8e,0x00]
vsetvli x8, x28, e32
# ALIAS:    vsetvli s0, t3, e32,m1          # encoding: [0x57,0x74,0x8e,0x00]
# NO-ALIAS: vsetvli s0, t3, e32,m1          # encoding: [0x57,0x74,0x8e,0x00]
vsetvli x8, x28, e32, m1

# ALIAS:    vwcvt.x.x.v     v2, v1          # encoding: [0x57,0x61,0x10,0xc6]
# NO-ALIAS: vwadd.vx        v2, v1, zero    # encoding: [0x57,0x61,0x10,0xc6]
vwcvt.x.x.v v2, v1
# ALIAS:    vwcvt.x.x.v     v2, v1, v0.t    # encoding: [0x57,0x61,0x10,0xc4]
# NO-ALIAS: vwadd.vx        v2, v1, zero, v0.t # encoding: [0x57,0x61,0x10,0xc4]
vwcvt.x.x.v v2, v1, v0.t
# ALIAS:    vwcvtu.x.x.v    v2, v1          # encoding: [0x57,0x61,0x10,0xc2]
# NO-ALIAS: vwaddu.vx       v2, v1, zero    # encoding: [0x57,0x61,0x10,0xc2]
vwcvtu.x.x.v v2, v1
# ALIAS:    vwcvtu.x.x.v    v2, v1, v0.t    # encoding: [0x57,0x61,0x10,0xc0]
# NO-ALIAS: vwaddu.vx       v2, v1, zero, v0.t # encoding: [0x57,0x61,0x10,0xc0]
vwcvtu.x.x.v v2, v1, v0.t
# ALIAS:    vnot.v  v0                      # encoding: [0x57,0xb0,0x0f,0x2e]
# NO-ALIAS: vxor.vi v0, v0, -1              # encoding: [0x57,0xb0,0x0f,0x2e]
vnot.v v0
# ALIAS:    vnot.v  v0, v0.t                # encoding: [0x57,0xb0,0x0f,0x2c]
# NO-ALIAS: vxor.vi v0, v0, -1, v0.t        # encoding: [0x57,0xb0,0x0f,0x2c]
vnot.v v0, v0.t
# ALIAS:    vmslt.vv        v0, v1, v0      # encoding: [0x57,0x00,0x10,0x6e]
# NO-ALIAS: vmslt.vv        v0, v1, v0      # encoding: [0x57,0x00,0x10,0x6e]
vmsgt.vv v0, v0, v1
# ALIAS:    vmslt.vv        v0, v1, v0, v0.t # encoding: [0x57,0x00,0x10,0x6c]
# NO-ALIAS: vmslt.vv        v0, v1, v0, v0.t # encoding: [0x57,0x00,0x10,0x6c]
vmsgt.vv v0, v0, v1, v0.t
# ALIAS:    vmsltu.vv       v0, v1, v0      # encoding: [0x57,0x00,0x10,0x6a]
# NO-ALIAS: vmsltu.vv       v0, v1, v0      # encoding: [0x57,0x00,0x10,0x6a]
vmsgtu.vv v0, v0, v1
# ALIAS:    vmsltu.vv       v0, v1, v0, v0.t # encoding: [0x57,0x00,0x10,0x68]
# NO-ALIAS: vmsltu.vv       v0, v1, v0, v0.t # encoding: [0x57,0x00,0x10,0x68]
vmsgtu.vv v0, v0, v1, v0.t
# ALIAS:    vmsle.vv        v0, v1, v0      # encoding: [0x57,0x00,0x10,0x76]
# NO-ALIAS: vmsle.vv        v0, v1, v0      # encoding: [0x57,0x00,0x10,0x76]
vmsge.vv v0, v0, v1
# ALIAS:    vmsle.vv        v0, v1, v0, v0.t # encoding: [0x57,0x00,0x10,0x74]
# NO-ALIAS: vmsle.vv        v0, v1, v0, v0.t # encoding: [0x57,0x00,0x10,0x74]
vmsge.vv v0, v0, v1, v0.t
# ALIAS:    vmsleu.vv       v0, v1, v0      # encoding: [0x57,0x00,0x10,0x72]
# NO-ALIAS: vmsleu.vv       v0, v1, v0      # encoding: [0x57,0x00,0x10,0x72]
vmsgeu.vv v0, v0, v1
# ALIAS:    vmsleu.vv       v0, v1, v0, v0.t # encoding: [0x57,0x00,0x10,0x70]
# NO-ALIAS: vmsleu.vv       v0, v1, v0, v0.t # encoding: [0x57,0x00,0x10,0x70]
vmsgeu.vv v0, v0, v1, v0.t
# ALIAS:    vmsle.vi        v0, v0, 15      # encoding: [0x57,0xb0,0x07,0x76]
# NO-ALIAS: vmsle.vi        v0, v0, 15      # encoding: [0x57,0xb0,0x07,0x76]
vmslt.vi v0, v0, 16
# ALIAS:    vmsle.vi        v0, v0, 15, v0.t # encoding: [0x57,0xb0,0x07,0x74]
# NO-ALIAS: vmsle.vi        v0, v0, 15, v0.t # encoding: [0x57,0xb0,0x07,0x74]
vmslt.vi v0, v0, 16, v0.t
# ALIAS:    vmsleu.vi       v0, v0, 15      # encoding: [0x57,0xb0,0x07,0x72]
# NO-ALIAS: vmsleu.vi       v0, v0, 15      # encoding: [0x57,0xb0,0x07,0x72]
vmsltu.vi v0, v0, 16
# ALIAS:    vmsleu.vi       v0, v0, 15, v0.t # encoding: [0x57,0xb0,0x07,0x70]
# NO-ALIAS: vmsleu.vi       v0, v0, 15, v0.t # encoding: [0x57,0xb0,0x07,0x70]
vmsltu.vi v0, v0, 16, v0.t
# ALIAS:    vmsgt.vi        v0, v0, 15      # encoding: [0x57,0xb0,0x07,0x7e]
# NO-ALIAS: vmsgt.vi        v0, v0, 15      # encoding: [0x57,0xb0,0x07,0x7e]
vmsge.vi v0, v0, 16
# ALIAS:    vmsgt.vi        v0, v0, 15, v0.t # encoding: [0x57,0xb0,0x07,0x7c]
# NO-ALIAS: vmsgt.vi        v0, v0, 15, v0.t # encoding: [0x57,0xb0,0x07,0x7c]
vmsge.vi v0, v0, 16, v0.t
# ALIAS:    vmsgtu.vi       v0, v0, 15      # encoding: [0x57,0xb0,0x07,0x7a]
# NO-ALIAS: vmsgtu.vi       v0, v0, 15      # encoding: [0x57,0xb0,0x07,0x7a]
vmsgeu.vi v0, v0, 16
# ALIAS:    vmsgtu.vi       v0, v0, 15, v0.t # encoding: [0x57,0xb0,0x07,0x78]
# NO-ALIAS: vmsgtu.vi       v0, v0, 15, v0.t # encoding: [0x57,0xb0,0x07,0x78]
vmsgeu.vi v0, v0, 16, v0.t
# ALIAS:    vmflt.vv        v0, v1, v0      # encoding: [0x57,0x10,0x10,0x6e]
# NO-ALIAS: vmflt.vv        v0, v1, v0      # encoding: [0x57,0x10,0x10,0x6e]
vmfgt.vv v0, v0, v1
# ALIAS:    vmflt.vv        v0, v1, v0, v0.t # encoding: [0x57,0x10,0x10,0x6c]
# NO-ALIAS: vmflt.vv        v0, v1, v0, v0.t # encoding: [0x57,0x10,0x10,0x6c]
vmfgt.vv v0, v0, v1, v0.t
# ALIAS:    vmfle.vv        v0, v1, v0      # encoding: [0x57,0x10,0x10,0x66]
# NO-ALIAS: vmfle.vv        v0, v1, v0      # encoding: [0x57,0x10,0x10,0x66]
vmfge.vv v0, v0, v1
# ALIAS:    vmfle.vv        v0, v1, v0, v0.t # encoding: [0x57,0x10,0x10,0x64]
# NO-ALIAS: vmfle.vv        v0, v1, v0, v0.t # encoding: [0x57,0x10,0x10,0x64]
vmfge.vv v0, v0, v1, v0.t
# ALIAS:    vmcpy.m v0, v1                  # encoding: [0x57,0xa0,0x10,0x66]
# NO-ALIAS: vmand.mm        v0, v1, v1      # encoding: [0x57,0xa0,0x10,0x66]
vmcpy.m v0, v1
# ALIAS:    vmclr.m v0                      # encoding: [0x57,0x20,0x00,0x6e]
# NO-ALIAS: vmxor.mm        v0, v0, v0      # encoding: [0x57,0x20,0x00,0x6e]
vmclr.m v0
# ALIAS:    vmset.m v0                      # encoding: [0x57,0x20,0x00,0x7e]
# NO-ALIAS: vmxnor.mm       v0, v0, v0      # encoding: [0x57,0x20,0x00,0x7e]
vmset.m v0
# ALIAS:    vmnot.m v0, v1                  # encoding: [0x57,0xa0,0x10,0x76]
# NO-ALIAS: vmnand.mm       v0, v1, v1      # encoding: [0x57,0xa0,0x10,0x76]
vmnot.m v0, v1

# Note: Here we check the immediate range boundaries for the alias.

# ALIAS:    vmsle.vi        v1, v2, -16     # encoding: [0xd7,0x30,0x28,0x76]
# NO-ALIAS: vmsle.vi        v1, v2, -16     # encoding: [0xd7,0x30,0x28,0x76]
vmslt.vi v1, v2, -15
# ALIAS:    vmsle.vi        v1, v2, -16, v0.t # encoding: [0xd7,0x30,0x28,0x74]
# NO-ALIAS: vmsle.vi        v1, v2, -16, v0.t # encoding: [0xd7,0x30,0x28,0x74]
vmslt.vi v1, v2, -15, v0.t
# ALIAS:    vmsle.vi        v1, v2, -1      # encoding: [0xd7,0xb0,0x2f,0x76]
# NO-ALIAS: vmsle.vi        v1, v2, -1      # encoding: [0xd7,0xb0,0x2f,0x76]
vmslt.vi v1, v2, 0
# ALIAS:    vmsle.vi        v1, v2, 15      # encoding: [0xd7,0xb0,0x27,0x76]
# NO-ALIAS: vmsle.vi        v1, v2, 15      # encoding: [0xd7,0xb0,0x27,0x76]
vmslt.vi v1, v2, 16
