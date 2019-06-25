//===- IntrinsicsRISCV.td - Defines EPI RISCV intrinsics ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the EPI RISCV-specific intrinsics.
//
//===----------------------------------------------------------------------===//

def llvm_epi_i8_ty : LLVMType<nxv1i8>;
def llvm_epi_i16_ty : LLVMType<nxv1i16>;
def llvm_epi_i32_ty : LLVMType<nxv1i32>;
def llvm_epi_i64_ty : LLVMType<nxv1i64>;

def llvm_epi_f32_ty : LLVMType<nxv1f32>;
def llvm_epi_f64_ty : LLVMType<nxv1f64>;

def llvm_ptr_i8_ty : LLVMPointerType<llvm_i8_ty>;
def llvm_ptr_i16_ty : LLVMPointerType<llvm_i16_ty>;
def llvm_ptr_i32_ty : LLVMPointerType<llvm_i32_ty>;
def llvm_ptr_i64_ty : LLVMPointerType<llvm_i64_ty>;

def llvm_ptr_f32_ty : LLVMPointerType<llvm_float_ty>;
def llvm_ptr_f64_ty : LLVMPointerType<llvm_double_ty>;

class EPIIntrinsic {
  // These intrinsics may accept illegal integer values in their llvm_any_ty
  // operand, so they have to be extended. If set to zero then the intrinsic
  // does not have any operand that must be extended.
  Intrinsic IntrinsicID = !cast<Intrinsic>(NAME);
  bits<4> ExtendOperand = 0;
  bits<4> MaskOperand = 0;
  bits<4> GVLOperand = 0;
}

let TargetPrefix = "epi" in {

def int_epi_vsetvl    : Intrinsic<[llvm_i64_ty], [
    /* requested vector length */  llvm_i64_ty,
    /* single element width */     llvm_i64_ty,
    /* vector length multiplier */ llvm_i64_ty], []>;
def int_epi_vsetvlmax : Intrinsic<[llvm_i64_ty], [
    /* single element width */     llvm_i64_ty,
    /* vector length multiplier */ llvm_i64_ty], []>;

class EPIBinary : Intrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_any_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 2;
  let GVLOperand = 3;
}
class EPIBinaryMask : Intrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                        llvm_anyvector_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 3;
  let MaskOperand = 4;
  let GVLOperand = 5;
}
class EPIBinaryMaskIn : Intrinsic< [ llvm_anyvector_ty ],
                        [ LLVMMatchType<0>, llvm_any_ty,
                          llvm_anyvector_ty, llvm_i64_ty ],
                        [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 2;
  let MaskOperand = 3;
  let GVLOperand = 4;
}
multiclass epi_binary {
  def "int_epi_" # NAME : EPIBinary;
  def "int_epi_" # NAME # "_mask" : EPIBinaryMask;
}
multiclass epi_binary_nomask {
  def "int_epi_" # NAME : EPIBinary;
}

multiclass epi_binary_mask_in {
  def "int_epi_" # NAME : EPIBinaryMaskIn;
}

class EPIBinaryWide : Intrinsic< [ llvm_anyvector_ty ],
                      [ llvm_anyvector_ty, llvm_any_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 2;
  let GVLOperand = 3;
}
class EPIBinaryWideMask : Intrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                        llvm_anyvector_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 3;
  let MaskOperand = 4;
  let GVLOperand = 5;
}
multiclass epi_binary_wide {
  def "int_epi_" # NAME : EPIBinaryWide;
  def "int_epi_" # NAME # "_mask" : EPIBinaryWideMask;
}

class EPIBinaryRelational : Intrinsic< [ llvm_anyvector_ty ],
                      [ llvm_anyvector_ty, llvm_any_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 2;
  let GVLOperand = 3;
}
class EPIBinaryRelationalMask : Intrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                        LLVMMatchType<0>, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 3;
  let MaskOperand = 4;
  let GVLOperand = 5;
}
multiclass epi_relational {
  def "int_epi_" # NAME : EPIBinaryRelational;
  def "int_epi_" # NAME # "_mask" : EPIBinaryRelationalMask;
}

class EPIUnary : Intrinsic< [ llvm_anyvector_ty ],
                      [ llvm_any_ty, llvm_i64_ty],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 1;
  let GVLOperand = 2;
}
class EPIUnaryMask : Intrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_any_ty, llvm_anyvector_ty,
                        llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 2;
  let MaskOperand = 3;
  let GVLOperand = 4;
}
multiclass epi_unary {
  def "int_epi_" # NAME : EPIUnary;
  def "int_epi_" # NAME # "_mask" : EPIUnaryMask;
}

class EPIUnaryScalar : Intrinsic< [ llvm_i64_ty ],
                      [ llvm_anyvector_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic;
class EPIUnaryScalarMask : Intrinsic< [ llvm_i64_ty ],
                      [ llvm_anyvector_ty, LLVMMatchType<0>, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic;
multiclass epi_unary_mask_to_scalar {
  def "int_epi_" # NAME : EPIUnaryScalar;
  def "int_epi_" # NAME # "_mask" : EPIUnaryScalarMask;
}

class EPITernary : Intrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                        llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 2;
  let GVLOperand = 4;
}
class EPITernaryMask : Intrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                        llvm_anyvector_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 2;
  let MaskOperand = 4;
  let GVLOperand = 5;
}
multiclass epi_ternary {
  def "int_epi_" # NAME : EPITernary;
  def "int_epi_" # NAME # "_mask" : EPITernaryMask;
}

class EPITernaryWide : Intrinsic< [ llvm_anyvector_ty ],
                      [ llvm_anyvector_ty, llvm_any_ty, LLVMMatchType<0>,
                        llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 2;
}
class EPITernaryWideMask : Intrinsic< [ llvm_anyvector_ty ],
                      [ llvm_anyvector_ty, llvm_any_ty, LLVMMatchType<0>,
                        llvm_anyvector_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 2;
}
multiclass epi_ternary_wide {
  def "int_epi_" # NAME : EPITernaryWide;
  def "int_epi_" # NAME # "_mask" : EPITernaryWideMask;
}

multiclass epi_unary_mask {
def int_epi_# NAME : Intrinsic<[ llvm_anyvector_ty ],
                               [ LLVMMatchType<0>, llvm_i64_ty ],
                               [ IntrNoMem ]>,
                     EPIIntrinsic;
def int_epi_# NAME #_mask : Intrinsic<[ llvm_anyvector_ty ],
                               [ LLVMMatchType<0>, LLVMMatchType<0>,
                                 llvm_i64_ty ],
                               [ IntrNoMem ]>,
                     EPIIntrinsic;
}

multiclass epi_conversions {
def int_epi_# NAME : Intrinsic<[ llvm_anyvector_ty ],
                               [ llvm_anyvector_ty, llvm_i64_ty ],
                               [ IntrNoMem ]>,
                     EPIIntrinsic;
def int_epi_# NAME #_mask : Intrinsic<[ llvm_anyvector_ty ],
                               [ LLVMMatchType<0>, llvm_anyvector_ty,
                                 llvm_anyvector_ty, llvm_i64_ty ],
                               [ IntrNoMem ]>,
                     EPIIntrinsic;
}


defm vadd : epi_binary;
defm vsub : epi_binary;
defm vrsub : epi_binary;

defm vwaddu : epi_binary_wide;
defm vwadd : epi_binary_wide;
defm vwaddu_w : epi_binary;
defm vwadd_w : epi_binary;
defm vwsubu : epi_binary_wide;
defm vwsub : epi_binary_wide;
defm vwsubu_w : epi_binary;
defm vwsub_w : epi_binary;

defm vadc : epi_ternary;
defm vmadc : epi_ternary;
defm vsbc : epi_ternary;
defm vmsbc : epi_ternary;

defm vand : epi_binary;
defm vor : epi_binary;
defm vxor : epi_binary;

defm vsll : epi_binary;
defm vsrl : epi_binary;
defm vsra : epi_binary;

// These actually do narrowing but type-wise they
// are like their widening counterparts
defm vnsrl : epi_binary_wide;
defm vnsra : epi_binary_wide;

defm vmseq : epi_relational;
defm vmsne : epi_relational;
defm vmsltu : epi_relational;
defm vmslt : epi_relational;
defm vmsleu : epi_relational;
defm vmsle : epi_relational;
defm vmsgtu : epi_relational;
defm vmsgt : epi_relational;

defm vminu : epi_binary;
defm vmin : epi_binary;
defm vmaxu : epi_binary;
defm vmax : epi_binary;

defm vmul : epi_binary;
defm vmulh : epi_binary;
defm vmulhu : epi_binary;
defm vmulhsu : epi_binary;

defm vwmul : epi_binary_wide;
defm vwmulu : epi_binary_wide;
defm vwmulsu : epi_binary_wide;

defm vmacc : epi_ternary;
defm vmsac : epi_ternary;
defm vmadd : epi_ternary;
defm vmsub : epi_ternary;

defm vwmaccu : epi_ternary_wide;
defm vwmacc : epi_ternary_wide;
defm vwmsacu : epi_ternary_wide;
defm vwmsac : epi_ternary_wide;

defm vdivu : epi_binary;
defm vdiv : epi_binary;
defm vremu : epi_binary;
defm vrem : epi_binary;

defm vmerge : epi_binary_mask_in;

defm vsaddu : epi_binary;
defm vsadd : epi_binary;
defm vssubu : epi_binary;
defm vssub : epi_binary;

defm vaadd : epi_binary;
defm vasub : epi_binary;

defm vsmul : epi_binary;

defm vwsmaccu : epi_ternary_wide;
defm vwsmacc : epi_ternary_wide;
defm vwsmsacu : epi_ternary_wide;
defm vwsmsac : epi_ternary_wide;

defm vssrl : epi_binary;
defm vssra : epi_binary;

defm vnclipu : epi_binary;
defm vnclip : epi_binary;

defm vfadd : epi_binary;
defm vfsub : epi_binary;
defm vfwadd : epi_binary_wide;
defm vfwsub : epi_binary_wide;
defm vfwadd_w : epi_binary;
defm vfwsub_w : epi_binary;

defm vfmul : epi_binary;
defm vfdiv : epi_binary;
defm vfrdiv : epi_binary;

defm vfwmul : epi_binary_wide;

defm vfmadd : epi_ternary;
defm vfnmadd : epi_ternary;
defm vfmsub : epi_ternary;
defm vfnmsub : epi_ternary;
defm vfmacc : epi_ternary;
defm vfnmacc : epi_ternary;
defm vfmsac : epi_ternary;
defm vfnmsac : epi_ternary;

defm vfwmacc : epi_ternary_wide;
defm vfwnmacc : epi_ternary_wide;
defm vfwmsac : epi_ternary_wide;
defm vfwnmsac : epi_ternary_wide;

defm vfmin : epi_binary;
defm vfmax : epi_binary;

defm vfsgnj : epi_binary;
defm vfsgnjn : epi_binary;
defm vfsgnjx : epi_binary;

defm vmfeq : epi_relational;
defm vmfne : epi_relational;
defm vmflt : epi_relational;
defm vmfle : epi_relational;
defm vmfgt : epi_relational;
defm vmfge : epi_relational;
defm vmford : epi_relational;

defm vfmerge : epi_binary_mask_in;

defm vredsum : epi_binary;
defm vredand : epi_binary;
defm vredor : epi_binary;
defm vredxor : epi_binary;
defm vredminu : epi_binary;
defm vredmin : epi_binary;
defm vredmaxu : epi_binary;
defm vredmax : epi_binary;

defm vwredsumu : epi_binary_wide;
defm vwredsum : epi_binary_wide;

defm vfredsum : epi_binary;
defm vfredosum : epi_binary;
defm vfredmin : epi_binary;
defm vfredmax : epi_binary;

defm vfwredsum : epi_binary_wide;
defm vfwredosum : epi_binary_wide;

defm vmandnot : epi_binary_nomask;
defm vmand : epi_binary_nomask;
defm vmor : epi_binary_nomask;
defm vmxor : epi_binary_nomask;
defm vmornot : epi_binary_nomask;
defm vmnand : epi_binary_nomask;
defm vmnor : epi_binary_nomask;
defm vmxnor : epi_binary_nomask;

defm vdotu : epi_binary;
defm vdot : epi_binary;

defm vfdot : epi_binary;

defm vcompress : epi_binary;

defm vrgather : epi_binary;

defm vmpopc : epi_unary_mask_to_scalar;
defm vmfirst : epi_unary_mask_to_scalar;

defm vslideup : epi_binary;
defm vslidedown : epi_binary;
defm vslide1up : epi_binary;
defm vslide1down : epi_binary;

def int_epi_vext_x_v : Intrinsic< [ llvm_anyint_ty ],
                      [ llvm_anyvector_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic;

def int_epi_vmv_s_x : Intrinsic< [ llvm_anyvector_ty ],
                      [ llvm_anyint_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 1;
}

def int_epi_vfmv_f_s : Intrinsic< [ llvm_anyfloat_ty ],
                                  [ llvm_anyvector_ty, llvm_i64_ty ],
                                  [ IntrNoMem ]>, EPIIntrinsic;
def int_epi_vfmv_s_f : Intrinsic< [ llvm_anyvector_ty ],
                                  [ llvm_anyfloat_ty, llvm_i64_ty ],
                                  [ IntrNoMem ]>, EPIIntrinsic;

defm vfsqrt : epi_unary;
defm vfclass : epi_unary;

defm vmsbf : epi_unary_mask;
defm vmsof : epi_unary_mask;
defm vmsif : epi_unary_mask;

def int_epi_viota : Intrinsic<[ llvm_anyvector_ty ],
                              [ llvm_anyvector_ty, llvm_i64_ty ],
                              [ IntrNoMem ]>,
                     EPIIntrinsic;
def int_epi_viota_mask : Intrinsic<[ llvm_anyvector_ty ],
                                   [ llvm_anyvector_ty, LLVMMatchType<1>,
                                     llvm_i64_ty ],
                                   [ IntrNoMem ]>,
                     EPIIntrinsic;

def int_epi_vid : Intrinsic< [ llvm_anyvector_ty ],
                             [ llvm_i64_ty ], [ IntrNoMem ]>,
                  EPIIntrinsic;
def int_epi_vid_mask : Intrinsic< [ llvm_anyvector_ty ],
                                  [ LLVMMatchType<0>, llvm_anyvector_ty,
                                    llvm_i64_ty ], [ IntrNoMem ]>,
                  EPIIntrinsic;

defm vfcvt_xu_f : epi_conversions;
defm vfcvt_x_f : epi_conversions;
defm vfcvt_f_xu : epi_conversions;
defm vfcvt_f_x : epi_conversions;

defm vfwcvt_xu_f : epi_conversions;
defm vfwcvt_x_f : epi_conversions;
defm vfwcvt_f_xu : epi_conversions;
defm vfwcvt_f_x : epi_conversions;
defm vfwcvt_f_f : epi_conversions;

defm vfncvt_xu_f : epi_conversions;
defm vfncvt_x_f : epi_conversions;
defm vfncvt_f_xu : epi_conversions;
defm vfncvt_f_x : epi_conversions;
defm vfncvt_f_f : epi_conversions;

defm vwcvt_xu_x : epi_conversions;
defm vwcvt_x_x : epi_conversions;

defm vncvt_x_x : epi_conversions;

def int_epi_vbroadcast : EPIUnary;

// Loads
multiclass epi_load {

  def ""              : Intrinsic<[ llvm_anyvector_ty ],
                                  [ LLVMPointerType<LLVMMatchType<0>>,
                                    llvm_i64_ty ],
                                  [ NoCapture<0>, IntrReadMem ]>;
  def "_strided"      : Intrinsic<[ llvm_anyvector_ty ],
                                  [ LLVMPointerType<LLVMMatchType<0>>,
                                    llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<0>, IntrReadMem ]>;
  def "_indexed"      : Intrinsic<[ llvm_anyvector_ty ],
                                  [ LLVMPointerType<LLVMMatchType<0>>,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<0>, IntrReadMem ]>;
  def "_mask"         : Intrinsic<[ llvm_anyvector_ty ],
                                  [ LLVMMatchType<0>,
                                    LLVMPointerType<LLVMMatchType<0>>,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<0>, IntrReadMem ]>;
  def "_strided_mask" : Intrinsic<[ llvm_anyvector_ty ],
                                  [ LLVMMatchType<0>,
                                    LLVMPointerType<LLVMMatchType<0>>,
                                    llvm_i64_ty, llvm_anyvector_ty,
                                    llvm_i64_ty ],
                                  [ NoCapture<0>, IntrReadMem ]>;
  def "_indexed_mask" : Intrinsic<[ llvm_anyvector_ty ],
                                  [ LLVMMatchType<0>,
                                    LLVMPointerType<LLVMMatchType<0>>,
                                    llvm_anyvector_ty, llvm_anyvector_ty,
                                    llvm_i64_ty ],
                                  [ NoCapture<0>, IntrReadMem ]>;
}

defm int_epi_vload : epi_load;

// Stores
multiclass epi_store {
  def ""              : Intrinsic<[],
                                  [ llvm_anyvector_ty,
                                    LLVMPointerType<LLVMMatchType<0>>,
                                    llvm_i64_ty ],
                                  [ NoCapture<1>, IntrWriteMem ]>;
  def "_strided"      : Intrinsic<[],
                                  [ llvm_anyvector_ty,
                                    LLVMPointerType<LLVMMatchType<0>>,
                                    llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<1>, IntrWriteMem ]>;
  def "_indexed"      : Intrinsic<[],
                                  [ llvm_anyvector_ty,
                                    LLVMPointerType<LLVMMatchType<0>>,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<1>, IntrWriteMem ]>;
  def "_mask"         : Intrinsic<[],
                                  [ llvm_anyvector_ty,
                                    LLVMPointerType<LLVMMatchType<0>>,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<1>, IntrWriteMem ]>;
  def "_strided_mask" : Intrinsic<[],
                                  [ llvm_anyvector_ty,
                                    LLVMPointerType<LLVMMatchType<0>>,
                                    llvm_i64_ty, llvm_anyvector_ty,
                                    llvm_i64_ty ],
                                  [ NoCapture<1>, IntrWriteMem ]>;
  def "_indexed_mask" : Intrinsic<[],
                                  [ llvm_anyvector_ty,
                                    LLVMPointerType<LLVMMatchType<0>>,
                                    llvm_anyvector_ty, llvm_anyvector_ty,
                                    llvm_i64_ty ],
                                  [ NoCapture<1>, IntrWriteMem ]>;
}

defm int_epi_vstore : epi_store;
defm int_epi_vstore_unordered : epi_store;

} // TargetPrefix = "epi"

