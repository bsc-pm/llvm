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

def llvm_epi_i1_ty : LLVMType<nxv1i1>;

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
  // operand, so they have to be extended. If set to all zeros then the
  // intrinsic does not have any operand that must be extended.
  Intrinsic IntrinsicID = !cast<Intrinsic>(NAME);
  bits<4> ExtendOperand = 0b0000;
}

let TargetPrefix = "epi" in {

def int_epi_readvl : Intrinsic<[llvm_i64_ty], [], [IntrNoMem]>;
def int_epi_setvl  : Intrinsic<[llvm_i64_ty], [
    /* requested vector length */  llvm_i64_ty,
    /* single element width */     llvm_i64_ty,
    /* vector length multiplier */ llvm_i64_ty], []>;

class EPIBinary : Intrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_any_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 0b0010;
}
class EPIBinaryMask : Intrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_any_ty, llvm_epi_i1_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 0b0010;
}
multiclass epi_binary {
  def "int_epi_" # NAME : EPIBinary;
  def "int_epi_" # NAME # "_mask" : EPIBinaryMask;
}
multiclass epi_binary_nomask {
  def "int_epi_" # NAME : EPIBinary;
}

class EPIBinaryRelational : Intrinsic< [ llvm_epi_i1_ty ],
                      [ llvm_anyvector_ty, llvm_any_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 0b0010;
}
class EPIBinaryRelationalMask : Intrinsic< [ llvm_epi_i1_ty ],
                      [ llvm_anyvector_ty, llvm_any_ty, llvm_epi_i1_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 0b0010;
}
multiclass epi_relational {
  def "int_epi_" # NAME : EPIBinaryRelational;
  def "int_epi_" # NAME # "_mask" : EPIBinaryRelationalMask;
}

class EPIUnary : Intrinsic< [ llvm_anyvector_ty ],
                      [ llvm_any_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 0b0001;
}
let ExtendOperand = 0b0001 in
class EPIUnaryMask : Intrinsic< [ llvm_anyvector_ty ],
                      [ llvm_any_ty, llvm_epi_i1_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 0b0001;
}
multiclass epi_unary {
  def "int_epi_" # NAME : EPIUnary;
  def "int_epi_" # NAME # "_mask" : EPIUnaryMask;
}

class EPIUnaryScalar : Intrinsic< [ llvm_i64_ty ],
                      [ llvm_epi_i1_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic;
class EPIUnaryScalarMask : Intrinsic< [ llvm_i64_ty ],
                      [ llvm_epi_i1_ty, llvm_epi_i1_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic;
multiclass epi_unary_mask_to_scalar {
  def "int_epi_" # NAME : EPIUnaryScalar;
  def "int_epi_" # NAME # "_mask" : EPIUnaryScalarMask;
}

class EPITernary : Intrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0> ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 0b0010;
}
class EPITernaryMask : Intrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                        llvm_epi_i1_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic
{
  let ExtendOperand = 0b0010;
}
multiclass epi_ternary {
  def "int_epi_" # NAME : EPITernary;
  def "int_epi_" # NAME # "_mask" : EPITernaryMask;
}

multiclass epi_unary_mask {
def int_epi_# NAME : Intrinsic<[ llvm_epi_i1_ty ],
                               [ llvm_epi_i1_ty ],
                               [ IntrNoMem ]>, 
                     EPIIntrinsic;
def int_epi_# NAME #_mask : Intrinsic<[ llvm_epi_i1_ty ],
                               [ llvm_epi_i1_ty, llvm_epi_i1_ty ],
                               [ IntrNoMem ]>, 
                     EPIIntrinsic;
}

multiclass epi_conversions {
def int_epi_# NAME : Intrinsic<[ llvm_anyvector_ty ],
                               [ llvm_anyvector_ty ],
                               [ IntrNoMem ]>, 
                     EPIIntrinsic;
def int_epi_# NAME #_mask : Intrinsic<[ llvm_anyvector_ty ],
                               [ llvm_anyvector_ty, llvm_epi_i1_ty ],
                               [ IntrNoMem ]>, 
                     EPIIntrinsic;
}


defm vadd : epi_binary;
defm vsub : epi_binary;
defm vrsub : epi_binary;

defm vwaddu : epi_binary;
defm vwadd : epi_binary;
defm vwaddu_w : epi_binary;
defm vwadd_w : epi_binary;
defm vwsubu : epi_binary;
defm vwsub : epi_binary;
defm vwsubu_w : epi_binary;
defm vwsub_w : epi_binary;

defm vadc : epi_ternary;
defm vsbc : epi_ternary;

defm vand : epi_binary;
defm vor : epi_binary;
defm vxor : epi_binary;

defm vsll : epi_binary;
defm vsrl : epi_binary;
defm vsra : epi_binary;

defm vnsrl : epi_binary;
defm vnsra : epi_binary;

defm vseq : epi_relational;
defm vsne : epi_relational;
defm vsltu : epi_relational;
defm vslt : epi_relational;
defm vsleu : epi_relational;
defm vsle : epi_relational;
defm vsgtu : epi_relational;
defm vsgt : epi_relational;

defm vminu : epi_binary;
defm vmin : epi_binary;
defm vmaxu : epi_binary;
defm vmax : epi_binary;

defm vmul : epi_binary;
defm vmulh : epi_binary;
defm vmulhu : epi_binary;
defm vmulhsu : epi_binary;

defm vwmul : epi_binary;
defm vwmulu : epi_binary;
defm vwmulsu : epi_binary;

defm vmacc : epi_ternary;
defm vmsac : epi_ternary;
defm vmadd : epi_ternary;
defm vmsub : epi_ternary;

defm vwmaccu : epi_binary;
defm vwmacc : epi_binary;
defm vwmsacu : epi_binary;
defm vwmsac : epi_binary;

defm vdivu : epi_binary;
defm vdiv : epi_binary;
defm vremu : epi_binary;
defm vrem : epi_binary;

defm vmerge : epi_binary;

defm vsaddu : epi_binary;
defm vsadd : epi_binary;
defm vssubu : epi_binary;
defm vssub : epi_binary;

defm vaadd : epi_binary;
defm vasub : epi_binary;

defm vsmul : epi_binary;

defm vwsmaccu : epi_binary;
defm vwsmacc : epi_binary;
defm vwsmsacu : epi_binary;
defm vwsmsac : epi_binary;

defm vssrl : epi_binary;
defm vssra : epi_binary;

defm vnclipu : epi_binary;
defm vnclip : epi_binary;

defm vfadd : epi_binary;
defm vfsub : epi_binary;
defm vfwadd : epi_binary;
defm vfwsub : epi_binary;
defm vfwadd_w : epi_binary;
defm vfwsub_w : epi_binary;

defm vfmul : epi_binary;
defm vfdiv : epi_binary;
defm vfrdiv : epi_binary;

defm vfwmul : epi_binary;

defm vfmadd : epi_ternary;
defm vfnmadd : epi_ternary;
defm vfmsub : epi_ternary;
defm vfnmsub : epi_ternary;
defm vfmacc : epi_ternary;
defm vfnmacc : epi_ternary;
defm vfmsac : epi_ternary;
defm vfnmsac : epi_ternary;

defm vfwmacc : epi_ternary;
defm vfwnmacc : epi_ternary;
defm vfwmsac : epi_ternary;
defm vfwnmsac : epi_ternary;

defm vfmin : epi_binary;
defm vfmax : epi_binary;

defm vfsgnj : epi_binary;
defm vfsgnjn : epi_binary;
defm vfsgnjx : epi_binary;

defm vfeq : epi_relational;
defm vfne : epi_relational;
defm vflt : epi_relational;
defm vflte : epi_relational;
defm vfgt : epi_relational;
defm vfgte : epi_relational;
defm vford : epi_relational;

defm vfmerge : epi_binary;

defm vredsum : epi_binary;
defm vredand : epi_binary;
defm vredor : epi_binary;
defm vredxor : epi_binary;
defm vredminu : epi_binary;
defm vredmin : epi_binary;
defm vredmaxu : epi_binary;
defm vredmax : epi_binary;

defm vwredsumu : epi_binary;
defm vwredsum : epi_binary;

defm vfredsum : epi_binary;
defm vfredosum : epi_binary;
defm vfredmin : epi_binary;
defm vfredmax : epi_binary;

defm vfwredsum : epi_binary;
defm vfwredosum : epi_binary;

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

def int_epi_vext_x_v : Intrinsic< [ llvm_i64_ty ],
                      [ llvm_anyvector_ty, llvm_any_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic;

def int_epi_vmv_s_x : Intrinsic< [ llvm_anyvector_ty ],
                      [ llvm_anyint_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic;

def int_epi_vfmv_f_s : Intrinsic< [ llvm_anyfloat_ty ], [ llvm_anyvector_ty ],
                                  [ IntrNoMem ]>, EPIIntrinsic;
def int_epi_vfmv_s_f : Intrinsic< [ llvm_anyvector_ty ], [ llvm_anyfloat_ty ],
                                  [ IntrNoMem ]>, EPIIntrinsic;

// TODO: Unary intrinsics

defm vfsqrt : epi_unary;
defm vfclass : epi_unary;

defm vmsbf : epi_unary_mask;
defm vmsof : epi_unary_mask;
defm vmsif : epi_unary_mask;

def int_epi_vmiota : Intrinsic<[ llvm_anyvector_ty ],
                               [ llvm_epi_i1_ty ],
                               [ IntrNoMem ]>,
                     EPIIntrinsic;
def int_epi_vmiota_mask : Intrinsic<[ llvm_anyvector_ty ],
                               [ llvm_epi_i1_ty, llvm_epi_i1_ty ],
                               [ IntrNoMem ]>,
                     EPIIntrinsic;

def int_epi_vid : Intrinsic< [ llvm_anyvector_ty ],
                             [ ], [ IntrNoMem ]>,
                  EPIIntrinsic;
def int_epi_vid_mask : Intrinsic< [ llvm_anyvector_ty ],
                             [ llvm_epi_i1_ty ], [ IntrNoMem ]>,
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


// FIXME: Load/store intrinsics are not generic yet.
// Loads
multiclass epi_load<LLVMType val_ty, LLVMType ptr_ty> {

  def "" : Intrinsic<[ val_ty ], [ ptr_ty ],
                     [ NoCapture<0>, IntrReadMem ]>;
  def "_strided" : Intrinsic<[ val_ty ], [ ptr_ty, llvm_i64_ty ],
                             [ NoCapture<0>, IntrReadMem ]>;
  def "_indexed" : Intrinsic<[ val_ty ], [ ptr_ty, llvm_epi_i64_ty ],
                             [ NoCapture<0>, IntrReadMem ]>;
  def "_mask" : Intrinsic<[ val_ty ], [ ptr_ty, llvm_epi_i1_ty ],
                     [ NoCapture<0>, IntrReadMem ]>;
  def "_strided_mask" : Intrinsic<[ val_ty ],
                             [ ptr_ty, llvm_i64_ty, llvm_epi_i1_ty ],
                             [ NoCapture<0>, IntrReadMem ]>;
  def "_indexed_mask" : Intrinsic<[ val_ty ],
                             [ ptr_ty, llvm_epi_i64_ty, llvm_epi_i1_ty ],
                             [ NoCapture<0>, IntrReadMem ]>;
}

defm int_epi_vload_i8  : epi_load<llvm_epi_i8_ty, llvm_ptr_i8_ty>;
defm int_epi_vload_i16 : epi_load<llvm_epi_i16_ty, llvm_ptr_i16_ty>;
defm int_epi_vload_i32 : epi_load<llvm_epi_i32_ty, llvm_ptr_i32_ty>;
defm int_epi_vload_i64 : epi_load<llvm_epi_i64_ty, llvm_ptr_i64_ty>;

defm int_epi_vload_u8  : epi_load<llvm_epi_i8_ty, llvm_ptr_i8_ty>;
defm int_epi_vload_u16 : epi_load<llvm_epi_i16_ty, llvm_ptr_i16_ty>;
defm int_epi_vload_u32 : epi_load<llvm_epi_i32_ty, llvm_ptr_i32_ty>;
defm int_epi_vload_u64 : epi_load<llvm_epi_i64_ty, llvm_ptr_i64_ty>;

defm int_epi_vload_f32 : epi_load<llvm_epi_f32_ty, llvm_ptr_f32_ty>;
defm int_epi_vload_f64 : epi_load<llvm_epi_f64_ty, llvm_ptr_f64_ty>;

// Stores
multiclass epi_store<LLVMType val_ty, LLVMType ptr_ty> {

  def "" : Intrinsic<[], [ val_ty,  ptr_ty ],
                     [ NoCapture<1>, IntrWriteMem ]>;
  def "_strided" : Intrinsic<[], [ val_ty,  ptr_ty, llvm_i64_ty ],
                             [ NoCapture<1>, IntrWriteMem ]>;
  def "_indexed" : Intrinsic<[], [ val_ty, ptr_ty, llvm_epi_i64_ty ],
                             [ NoCapture<1>, IntrWriteMem ]>;
  def "_mask" : Intrinsic<[], [ val_ty,  ptr_ty, llvm_epi_i1_ty ],
                     [ NoCapture<1>, IntrWriteMem ]>;
  def "_strided_mask" : Intrinsic<[],
                             [ val_ty,  ptr_ty, llvm_i64_ty, llvm_epi_i1_ty ],
                             [ NoCapture<1>, IntrWriteMem ]>;
  def "_indexed_mask" : Intrinsic<[],
                             [ val_ty, ptr_ty, llvm_epi_i64_ty, llvm_epi_i1_ty ],
                             [ NoCapture<1>, IntrWriteMem ]>;
}

defm int_epi_vstore_i8  : epi_store<llvm_epi_i8_ty, llvm_ptr_i8_ty>;
defm int_epi_vstore_i16 : epi_store<llvm_epi_i16_ty, llvm_ptr_i16_ty>;
defm int_epi_vstore_i32 : epi_store<llvm_epi_i32_ty, llvm_ptr_i32_ty>;
defm int_epi_vstore_i64 : epi_store<llvm_epi_i64_ty, llvm_ptr_i64_ty>;

defm int_epi_vstore_unordered_i8  : epi_store<llvm_epi_i8_ty, llvm_ptr_i8_ty>;
defm int_epi_vstore_unordered_i16 : epi_store<llvm_epi_i16_ty, llvm_ptr_i16_ty>;
defm int_epi_vstore_unordered_i32 : epi_store<llvm_epi_i32_ty, llvm_ptr_i32_ty>;
defm int_epi_vstore_unordered_i64 : epi_store<llvm_epi_i64_ty, llvm_ptr_i64_ty>;

defm int_epi_vstore_f32 : epi_store<llvm_epi_f32_ty, llvm_ptr_f32_ty>;
defm int_epi_vstore_f64 : epi_store<llvm_epi_f64_ty, llvm_ptr_f64_ty>;

} // TargetPrefix = "epi"

