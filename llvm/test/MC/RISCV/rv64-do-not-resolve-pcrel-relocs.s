# RUN: llvm-mc -triple riscv64  -filetype=obj < %s | llvm-objdump  -r  -

# CHECK: 000000000000000c R_RISCV_PCREL_HI20 my_handler
# CHECK: 0000000000000010 R_RISCV_PCREL_LO12_I .Lpcrel_hi0

	.text
	.option	pic
	.file	"t.c"
	.globl	main                    # -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   # @main
# %bb.0:                                # %entry
	addi	sp, sp, -16
	sd	ra, 8(sp)
	addi	a0, zero, 2
	lla	a1, my_handler
	call	signal@plt
	mv	a0, zero
	ld	ra, 8(sp)
	addi	sp, sp, 16
	ret
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
                                        # -- End function
	.p2align	2               # -- Begin function my_handler
	.type	my_handler,@function
my_handler:                             # @my_handler
# %bb.0:                                # %entry
	addi	sp, sp, -16
	sd	ra, 8(sp)
	mv	a1, a0
	lla	a0, .L.str
	call	printf@plt
	ld	ra, 8(sp)
	addi	sp, sp, 16
	ret
.Lfunc_end1:
	.size	my_handler, .Lfunc_end1-my_handler
                                        # -- End function
	.type	.L.str,@object          # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Handler run! arg=%d\n"
	.size	.L.str, 21


	.ident	"clang version 7.0.0 (https://pm.bsc.es/gitlab/LLVM/clang 556681de171148046397f810e8228a4bbc8ab96a) (https://pm.bsc.es/gitlab/LLVM/llvm f357686b021c41fbc4532daa378c71cb3a47a867)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym my_handler
