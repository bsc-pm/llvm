//=- RISCVInstrInfoV.td - Zeou-Extension RISCV instructions -*- tblgen-*----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

include "RISCVInstrFormatsEPI.td"

//===----------------------------------------------------------------------===//
// Target Specific DAG nodes
//===----------------------------------------------------------------------===//

// FIXME: We should constraint the second operand
def SDTUnaryVectorFromScalar : SDTypeProfile<1, 1, [SDTCisVec<0>]>;

def riscv_vbroadcast : SDNode<"RISCVISD::VBROADCAST", SDTUnaryVectorFromScalar>;

//===----------------------------------------------------------------------===//

class VMask<bits<1> vtype>
{
  bits<1> Value = vtype;
}

def vmask_all_lanes : VMask<0b1>;
def vmask_only_true : VMask<0b0>;

class EPILookupIntrinsic<string basename> {
  Intrinsic I = !cast<Intrinsic>("int_epi_" # basename);
}

// FIXME: some of these simmN can be removed when things settle

def simm5 : Operand<XLenVT>, ImmLeaf<XLenVT, [{return isInt<5>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<5>;
  let DecoderMethod = "decodeSImmOperand<5>";
}

def VectorMaskOp : AsmOperandClass {
  let Name = "VectorMask";
  let RenderMethod = "addVectorMaskOperands";
  let DiagnosticType = "InvalidVectorMaskOperand";
}

def vectormask : RegisterOperand<EPIMaskVR, "printVectorMask"> {
  let ParserMatchClass = VectorMaskOp;
  let DecoderMethod = "decodeVectorMask";
}

def VectorElementWidthOp : AsmOperandClass {
  let Name = "VectorElementWidth";
  let RenderMethod = "addVectorElementWidthOperands";
  let DiagnosticType = "InvalidVectorElementWidthOperand";
}

def vectorelementwidth : Operand<XLenVT> {
  let ParserMatchClass = VectorElementWidthOp;
  let PrintMethod = "printVectorElementWidth";
  let DecoderMethod = "decodeVectorElementWidth";
}

def VectorMultiplierOp : AsmOperandClass {
  let Name = "VectorMultiplier";
  let RenderMethod = "addVectorMultiplierOperands";
  let DiagnosticType = "InvalidVectorMultiplierOperand";
}

def vectormultiplier : Operand<XLenVT> {
  let ParserMatchClass = VectorMultiplierOp;
  let PrintMethod = "printVectorMultiplier";
  let DecoderMethod = "decodeVectorMultiplier";
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Join strings in list using separator and ignoring empty elements
class Join<list<string> strings, string separator> {
  string ret = !foldl(!head(strings), !tail(strings), a, b,
                      !cond(
                        !and(!empty(a), !empty(b)) : "",
                        !empty(a) : b,
                        !empty(b) : a,
                        1 : a#separator#b));
}

//===----------------------------------------------------------------------===//
// Common definitions
//===----------------------------------------------------------------------===//

multiclass binary_mask<string opcodestr, bits<6> funct6> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE] in
  def _MM : RVInstEPIOPV_MVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, EPIVR:$rs1),
             opcodestr # ".mm",
             "${rd}, ${rs2}, ${rs1}">;
}

multiclass binary_arithmetic_integer_n_v<string opcodestr, bits<6> funct6,
  string destoperandkind> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE] in
  def V : RVInstEPIOPV_IVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, EPIVR:$rs1),
             opcodestr # "." # destoperandkind # "v",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def V_MASK : RVInstEPIOPV_IVV<funct6,
               (outs EPIVR:$rd),
               (ins EPIVR:$rs2, EPIVR:$rs1, vectormask:$vm),
               opcodestr # "." # destoperandkind # "v",
               "${rd}, ${rs2}, ${rs1}, ${vm}">;
}

multiclass binary_arithmetic_integer_n_x<string opcodestr, bits<6> funct6,
  string destoperandkind> {

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE] in
  def X : RVInstEPIOPV_IVX<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, GPR:$rs1),
             opcodestr # "." # destoperandkind # "x",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def X_MASK : RVInstEPIOPV_IVX<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, GPR:$rs1, vectormask:$vm),
             opcodestr # "." # destoperandkind # "x",
             "${rd}, ${rs2}, ${rs1}, ${vm}">;
}

multiclass binary_arithmetic_integer_n_i<string opcodestr, bits<6> funct6,
  string destoperandkind> {

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE] in
  def I : RVInstEPIOPV_IVI<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, simm5:$imm5),
             opcodestr # "." # destoperandkind # "i",
             "${rd}, ${rs2}, ${imm5}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def I_MASK : RVInstEPIOPV_IVI<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, simm5:$imm5, vectormask:$vm),
             opcodestr # "." # destoperandkind # "i",
             "${rd}, ${rs2}, ${imm5}, ${vm}">;
}

multiclass binary_arithmetic_integer_n_u<string opcodestr, bits<6> funct6,
  string destoperandkind> {

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE] in
  def U : RVInstEPIOPV_IVI<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, uimm5:$imm5),
             opcodestr # "." # destoperandkind # "i",
             "${rd}, ${rs2}, ${imm5}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def U_MASK : RVInstEPIOPV_IVI<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, uimm5:$imm5, vectormask:$vm),
             opcodestr # "." # destoperandkind # "i",
             "${rd}, ${rs2}, ${imm5}, ${vm}">;
}

multiclass binary_arithmetic_integer_n_s<string opcodestr, bits<6> funct6,
  string destoperandkind> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE] in
  def S : RVInstEPIOPV_IVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, EPIVR:$rs1),
             opcodestr # "." # destoperandkind # "s",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def S_MASK : RVInstEPIOPV_IVV<funct6,
               (outs EPIVR:$rd),
               (ins EPIVR:$rs2, EPIVR:$rs1, vectormask:$vm),
               opcodestr # "." # destoperandkind # "s",
               "${rd}, ${rs2}, ${rs1}, ${vm}">;
}

multiclass binary_arithmetic_integer_s<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_integer_n_s<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_integer_vxi<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_integer_n_v<opcodestr, funct6, "v">;
  defm _V : binary_arithmetic_integer_n_x<opcodestr, funct6, "v">;
  defm _V : binary_arithmetic_integer_n_i<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_integer_vxu<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_integer_n_v<opcodestr, funct6, "v">;
  defm _V : binary_arithmetic_integer_n_x<opcodestr, funct6, "v">;
  defm _V : binary_arithmetic_integer_n_u<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_integer_vx<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_integer_n_v<opcodestr, funct6, "v">;
  defm _V : binary_arithmetic_integer_n_x<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_integer_xi<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_integer_n_x<opcodestr, funct6, "v">;
  defm _V : binary_arithmetic_integer_n_i<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_integer_xu<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_integer_n_x<opcodestr, funct6, "v">;
  defm _V : binary_arithmetic_integer_n_u<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_integer_v<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_integer_n_v<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_integer_vx_mask_in<string opcodestr,
                                                bits<6> funct6,
                                                VMask vmask> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask.Value,
      Uses = [VL, VTYPE] in
  def _VVM : RVInstEPIOPV_IVV<funct6,
              (outs EPIVR:$rd),
              (ins EPIVR:$rs2, EPIVR:$rs1, EPIMaskVR:$mask),
              opcodestr # ".vvm",
              "${rd}, ${rs2}, ${rs1}, v0">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask.Value,
      Uses = [VL, VTYPE] in
  def _VXM : RVInstEPIOPV_IVX<funct6,
              (outs EPIVR:$rd),
              (ins EPIVR:$rs2, GPR:$rs1, EPIMaskVR:$mask),
              opcodestr # ".vxm",
              "${rd}, ${rs2}, ${rs1}, v0">;
}

multiclass binary_arithmetic_integer_vxi_mask_in<string opcodestr,
                                                 bits<6> funct6,
                                                 VMask vmask> {
  defm "" : binary_arithmetic_integer_vx_mask_in<opcodestr, funct6, vmask>;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask.Value,
      Uses = [VL, VTYPE] in
  def _VIM : RVInstEPIOPV_IVI<funct6,
              (outs EPIVR:$rd),
              (ins EPIVR:$rs2, simm5:$imm5, EPIMaskVR:$mask),
              opcodestr # ".vim",
              "${rd}, ${rs2}, ${imm5}, v0">;
}

multiclass binary_arithmetic_integer_vxi_mask<string opcodestr, bits<6> funct6> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def V_MASK : RVInstEPIOPV_IVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, EPIVR:$rs1, vectormask:$vm),
             opcodestr # ".vv",
             "${rd}, ${rs2}, ${rs1}, ${vm}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def X_MASK : RVInstEPIOPV_IVX<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, GPR:$rs1, vectormask:$vm),
             opcodestr # ".vx",
             "${rd}, ${rs2}, ${rs1}, ${vm}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def I_MASK : RVInstEPIOPV_IVI<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, simm5:$imm5, vectormask:$vm),
             opcodestr # ".vi",
             "${rd}, ${rs2}, ${imm5}, ${vm}">;
}

multiclass binary_arithmetic_float_n_v<string opcodestr, bits<6> funct6,
  string destoperandkind> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE] in
  def V : RVInstEPIOPV_FVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, EPIVR:$rs1),
             opcodestr # "." # destoperandkind # "v",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def V_MASK : RVInstEPIOPV_FVV<funct6,
               (outs EPIVR:$rd),
               (ins EPIVR:$rs2, EPIVR:$rs1, vectormask:$vm),
               opcodestr # "." # destoperandkind # "v",
               "${rd}, ${rs2}, ${rs1}, ${vm}">;
}

multiclass binary_arithmetic_float_n_s<string opcodestr, bits<6> funct6,
  string destoperandkind> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE] in
  def S : RVInstEPIOPV_FVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, EPIVR:$rs1),
             opcodestr # "." # destoperandkind # "s",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def S_MASK : RVInstEPIOPV_FVV<funct6,
               (outs EPIVR:$rd),
               (ins EPIVR:$rs2, EPIVR:$rs1, vectormask:$vm),
               opcodestr # "." # destoperandkind # "s",
               "${rd}, ${rs2}, ${rs1}, ${vm}">;
}

multiclass binary_arithmetic_float_n_f<string opcodestr, bits<6> funct6,
  string destoperandkind> {

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE] in
  def F : RVInstEPIOPV_FVF<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, FPR64:$rs1),
             opcodestr # "." # destoperandkind # "f",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def F_MASK : RVInstEPIOPV_FVF<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, FPR64:$rs1, vectormask:$vm),
             opcodestr # "." # destoperandkind # "f",
             "${rd}, ${rs2}, ${rs1}, ${vm}">;
}

multiclass binary_arithmetic_float_vf<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_float_n_v<opcodestr, funct6, "v">;
  defm _V : binary_arithmetic_float_n_f<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_float_wf<string opcodestr, bits<6> funct6> {
  defm _W : binary_arithmetic_float_n_v<opcodestr, funct6, "w">;
  defm _W : binary_arithmetic_float_n_f<opcodestr, funct6, "w">;
}

multiclass binary_arithmetic_float_vf_destructive<string opcodestr, bits<6> funct6> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Constraints = "$rd = $_rd", Uses = [VL, VTYPE] in
  def _VV : RVInstEPIOPV_FVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$_rd, EPIVR:$rs1, EPIVR:$rs2),
             opcodestr # ".vv",
             "${rd}, ${rs1}, ${rs2}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Constraints = "$rd = $_rd", Uses = [VL, VTYPE] in
  def _VV_MASK : RVInstEPIOPV_FVV<funct6,
               (outs EPIVR:$rd),
               (ins EPIVR:$_rd, EPIVR:$rs1, EPIVR:$rs2, vectormask:$vm),
               opcodestr # ".vv",
               "${rd}, ${rs1}, ${rs2}, ${vm}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Constraints = "$rd = $_rd", Uses = [VL, VTYPE] in
  def _VF : RVInstEPIOPV_FVF<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$_rd, FPR64:$rs1, EPIVR:$rs2),
             opcodestr # ".vf",
             "${rd}, ${rs1}, ${rs2}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Constraints = "$rd = $_rd", Uses = [VL, VTYPE] in
  def _VF_MASK : RVInstEPIOPV_FVF<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$_rd, FPR64:$rs1, EPIVR:$rs2, vectormask:$vm),
             opcodestr # ".vf",
             "${rd}, ${rs1}, ${rs2}, ${vm}">;
}

multiclass binary_arithmetic_float_f<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_float_n_f<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_float_v<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_float_n_v<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_float_vs<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_float_n_s<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_mask_integer_n_v<string opcodestr, bits<6> funct6,
  string destoperandkind> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE] in
  def V : RVInstEPIOPV_MVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, EPIVR:$rs1),
             opcodestr # "." # destoperandkind # "v",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def V_MASK : RVInstEPIOPV_MVV<funct6,
               (outs EPIVR:$rd),
               (ins EPIVR:$rs2, EPIVR:$rs1, vectormask:$vm),
               opcodestr # "." # destoperandkind # "v",
               "${rd}, ${rs2}, ${rs1}, ${vm}">;
}

multiclass binary_arithmetic_mask_integer_n_x<string opcodestr, bits<6> funct6,
  string destoperandkind> {

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE] in
  def X : RVInstEPIOPV_MVX<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, GPR:$rs1),
             opcodestr # "." # destoperandkind # "x",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def X_MASK : RVInstEPIOPV_MVX<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, GPR:$rs1, vectormask:$vm),
             opcodestr # "." # destoperandkind # "x",
             "${rd}, ${rs2}, ${rs1}, ${vm}">;
}

multiclass binary_arithmetic_mask_integer_v<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_mask_integer_n_v<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_mask_integer_n_s<string opcodestr, bits<6> funct6,
  string destoperandkind> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE] in
  def S : RVInstEPIOPV_MVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, EPIVR:$rs1),
             opcodestr # "." # destoperandkind # "s",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in
  def S_MASK : RVInstEPIOPV_MVV<funct6,
               (outs EPIVR:$rd),
               (ins EPIVR:$rs2, EPIVR:$rs1, vectormask:$vm),
               opcodestr # "." # destoperandkind # "s",
               "${rd}, ${rs2}, ${rs1}, ${vm}">;
}

multiclass binary_arithmetic_mask_integer_s<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_mask_integer_n_s<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_mask_integer_x<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_mask_integer_n_x<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_mask_integer_vx<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_mask_integer_n_v<opcodestr, funct6, "v">;
  defm _V : binary_arithmetic_mask_integer_n_x<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_mask_integer_wx<string opcodestr, bits<6> funct6> {
  defm _W : binary_arithmetic_mask_integer_n_v<opcodestr, funct6, "w">;
  defm _W : binary_arithmetic_mask_integer_n_x<opcodestr, funct6, "w">;
}

multiclass binary_arithmetic_mask_integer_vx_destructive<string opcodestr, bits<6> funct6,
  string destoperandkind> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Constraints = "$rd = $_rd", Uses = [VL, VTYPE] in
  def _VV : RVInstEPIOPV_MVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$_rd, EPIVR:$rs1, EPIVR:$rs2),
             opcodestr # "." # destoperandkind # "v",
             "${rd}, ${rs1}, ${rs2}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Constraints = "$rd = $_rd", Uses = [VL, VTYPE] in
  def _VV_MASK : RVInstEPIOPV_MVV<funct6,
               (outs EPIVR:$rd),
               (ins EPIVR:$_rd, EPIVR:$rs1, EPIVR:$rs2, vectormask:$vm),
               opcodestr # "." # destoperandkind # "v",
               "${rd}, ${rs1}, ${rs2}, ${vm}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Constraints = "$rd = $_rd", Uses = [VL, VTYPE] in
  def _VX : RVInstEPIOPV_MVX<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$_rd, GPR:$rs1, EPIVR:$rs2),
             opcodestr # "." # destoperandkind # "x",
             "${rd}, ${rs1}, ${rs2}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Constraints = "$rd = $_rd", Uses = [VL, VTYPE] in
  def _VX_MASK : RVInstEPIOPV_MVX<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$_rd, GPR:$rs1, EPIVR:$rs2, vectormask:$vm),
             opcodestr # "." # destoperandkind # "x",
             "${rd}, ${rs1}, ${rs2}, ${vm}">;
}

multiclass unary_arithmetic_nomask_integer_vxi<string opcodestr,
                                               bits<6> funct6> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE], rs2 = 0b00000 in
  def _V : RVInstEPIOPV_IVV<funct6,
            (outs EPIVR:$rd),
            (ins EPIVR:$rs1),
            opcodestr # ".v",
            "${rd}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE], rs2 = 0b00000 in
  def _X : RVInstEPIOPV_IVX<funct6,
            (outs EPIVR:$rd),
            (ins GPR:$rs1),
            opcodestr # ".x",
            "${rd}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE], rs2 = 0b00000 in
  def _I : RVInstEPIOPV_IVI<funct6,
            (outs EPIVR:$rd),
            (ins simm5:$imm5),
            opcodestr # ".i",
            "${rd}, ${imm5}">;
}

multiclass unary_mask_integer_m<string opcodestr, bits<6> funct6> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE], rs1 = 0b00000 in
  def _M : RVInstEPIOPV_MVV<funct6,
             (outs GPR:$rd),
             (ins EPIVR:$rs2),
             opcodestr # ".m",
             "${rd}, ${rs2}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE], rs1 = 0b00000 in
  def _M_MASK : RVInstEPIOPV_MVV<funct6,
               (outs GPR:$rd),
               (ins EPIVR:$rs2, vectormask:$vm),
               opcodestr # ".m",
               "${rd}, ${rs2}, ${vm}">;
}

multiclass unary_arithmetic_float_v<string opcodestr, bits<6> funct6,
                                    bits<5> rs1>
{
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE], rs1 = rs1 in
  def _V : RVInstEPIOPV_FVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2),
             opcodestr # ".v",
             "${rd}, ${rs2}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE], rs1 = rs1 in
  def _V_MASK : RVInstEPIOPV_FVV<funct6,
               (outs EPIVR:$rd),
               (ins EPIVR:$rs2, vectormask:$vm),
               opcodestr # ".v",
               "${rd}, ${rs2}, ${vm}">;
}

multiclass vector_store_unit_stride_width<string opcodestr, bits<3> width>
{
  let nf=0b000, rs2 = 0b00000, width = width,
      mayStore = 1, mayLoad = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in {
    let vm = vmask_all_lanes.Value in
    def _V : RVInstStore<0b000,
               (outs),
               (ins EPIVR:$rd, GPR:$rs1),
               opcodestr # ".v",
               "${rd}, (${rs1})">;

    let Uses = [VL, VTYPE] in
    def _V_MASK : RVInstStore<0b000,
                    (outs),
                    (ins EPIVR:$rd, GPR:$rs1, vectormask:$vm),
                    opcodestr # ".v",
                    "${rd}, (${rs1}), ${vm}">;
  }
}

multiclass vector_store_unit_stride<string opcodestr>
{
  defm B : vector_store_unit_stride_width<opcodestr # "b", 0b000>;
  defm H : vector_store_unit_stride_width<opcodestr # "h", 0b101>;
  defm W : vector_store_unit_stride_width<opcodestr # "w", 0b110>;
  defm E : vector_store_unit_stride_width<opcodestr # "e", 0b111>;
}

multiclass vector_store_nonunit_stride_width<string opcodestr, bits<3> mop,
                                             RegisterClass rc, bits<3> width>
{
  let nf=0b000, width = width,
      mayStore = 1, mayLoad = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE] in {
    let vm = vmask_all_lanes.Value in
    def _V : RVInstStore<mop,
               (outs),
               (ins EPIVR:$rd, GPR:$rs1, rc:$rs2),
               opcodestr # ".v",
               "${rd}, (${rs1}), ${rs2}">;

    let Uses = [VL, VTYPE] in
    def _V_MASK : RVInstStore<mop,
                    (outs),
                    (ins EPIVR:$rd, GPR:$rs1, rc:$rs2, vectormask:$vm),
                    opcodestr # ".v",
                    "${rd}, (${rs1}), ${rs2}, ${vm}">;
  }
}

multiclass vector_store_nonunit_stride<string opcodestr, bits<3> mop, RegisterClass rc>
{
  defm B : vector_store_nonunit_stride_width<opcodestr # "b", mop, rc, 0b000>;
  defm H : vector_store_nonunit_stride_width<opcodestr # "h", mop, rc, 0b101>;
  defm W : vector_store_nonunit_stride_width<opcodestr # "w", mop, rc, 0b110>;
  defm E : vector_store_nonunit_stride_width<opcodestr # "e", mop, rc, 0b111>;
}

multiclass vector_store<string opcodestr>
{
  defm "" : vector_store_unit_stride<opcodestr>;
  defm S : vector_store_nonunit_stride<opcodestr # "s",   0b010, GPR>;
  defm X : vector_store_nonunit_stride<opcodestr # "x",   0b011, EPIVR>;
  defm UX : vector_store_nonunit_stride<opcodestr # "ux", 0b111, EPIVR>;
}

multiclass vector_load_unit_stride_width_sign<string opcodestr, bits<1> sign,
                                              bits<3> width>
{
  let nf=0b000, rs2 = 0b00000, width = width,
      mayStore = 0, mayLoad = 1, hasSideEffects = 0,
      Uses = [VL, VTYPE] in {
    let vm = vmask_all_lanes.Value in
    def _V : RVInstLoad<{sign, 0b00},
               (outs EPIVR:$rd),
               (ins GPR:$rs1),
               opcodestr # ".v",
               "${rd}, (${rs1})">;

    let Uses = [VL, VTYPE] in
    def _V_MASK : RVInstLoad<{sign, 0b00},
                    (outs EPIVR:$rd),
                    (ins GPR:$rs1, vectormask:$vm),
                    opcodestr # ".v",
                    "${rd}, (${rs1}), ${vm}">;
  }
}

multiclass vector_load_unit_stride<string opcodestr>
{
  defm B : vector_load_unit_stride_width_sign<opcodestr # "b", 0b1, 0b000>;
  defm H : vector_load_unit_stride_width_sign<opcodestr # "h", 0b1, 0b101>;
  defm W : vector_load_unit_stride_width_sign<opcodestr # "w", 0b1, 0b110>;

  defm BU : vector_load_unit_stride_width_sign<opcodestr # "bu", 0b0, 0b000>;
  defm HU : vector_load_unit_stride_width_sign<opcodestr # "hu", 0b0, 0b101>;
  defm WU : vector_load_unit_stride_width_sign<opcodestr # "wu", 0b0, 0b110>;

  defm E : vector_load_unit_stride_width_sign<opcodestr # "e", 0b0, 0b111>;
}

multiclass vector_load_nonunit_stride_width_sign<string opcodestr, bits<1> sign,
                                      bits<2> mop, RegisterClass rc,
                                      bits<3> width>
{
  let nf=0b000, width = width,
      mayStore = 0, mayLoad = 1, hasSideEffects = 0,
      Uses = [VL, VTYPE] in {
    let vm = vmask_all_lanes.Value in
    def _V : RVInstLoad<{sign, mop},
               (outs EPIVR:$rd),
               (ins GPR:$rs1, rc:$rs2),
               opcodestr # ".v",
               "${rd}, (${rs1}), ${rs2}">;

    let Uses = [VL, VTYPE] in
    def _V_MASK : RVInstLoad<{sign, mop},
                    (outs EPIVR:$rd),
                    (ins GPR:$rs1, rc:$rs2, vectormask:$vm),
                    opcodestr # ".v",
                    "${rd}, (${rs1}), ${rs2}, ${vm}">;
  }
}

multiclass vector_load_nonunit_stride<string opcodestr, bits<2> mop,
                                      RegisterClass rc>
{
  defm B : vector_load_nonunit_stride_width_sign<opcodestr # "b", 0b1, mop, rc, 0b000>;
  defm H : vector_load_nonunit_stride_width_sign<opcodestr # "h", 0b1, mop, rc, 0b101>;
  defm W : vector_load_nonunit_stride_width_sign<opcodestr # "w", 0b1, mop, rc, 0b110>;

  defm BU : vector_load_nonunit_stride_width_sign<opcodestr # "bu", 0b0, mop, rc, 0b000>;
  defm HU : vector_load_nonunit_stride_width_sign<opcodestr # "hu", 0b0, mop, rc, 0b101>;
  defm WU : vector_load_nonunit_stride_width_sign<opcodestr # "wu", 0b0, mop, rc, 0b110>;

  defm E : vector_load_nonunit_stride_width_sign<opcodestr # "e", 0b0, mop, rc, 0b111>;
}

multiclass vector_load<string opcodestr>
{
  defm "" : vector_load_unit_stride<opcodestr>;
  defm S : vector_load_nonunit_stride<opcodestr # "s",   0b10, GPR>;
  defm X : vector_load_nonunit_stride<opcodestr # "x",   0b11, EPIVR>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasExtEPI] in {

defm VADD  : binary_arithmetic_integer_vxi<"vadd", 0b000000>;

defm VSUB  : binary_arithmetic_integer_vx<"vsub",  0b000010>;
defm VRSUB : binary_arithmetic_integer_xi<"vrsub", 0b000011>;

defm VMINU : binary_arithmetic_integer_vx<"vminu", 0b000100>;
defm VMIN  : binary_arithmetic_integer_vx<"vmin",  0b000101>;

defm VMAXU : binary_arithmetic_integer_vx<"vmaxu", 0b000110>;
defm VMAX  : binary_arithmetic_integer_vx<"vmax",  0b000111>;

defm VAND  : binary_arithmetic_integer_vxi<"vand", 0b001001>;
defm VOR   : binary_arithmetic_integer_vxi<"vor", 0b001010>;
defm VXOR  : binary_arithmetic_integer_vxi<"vxor", 0b001011>;

defm VRGATHER : binary_arithmetic_integer_vxi<"vrgather", 0b001100>;

defm VSLIDEUP   : binary_arithmetic_integer_xu<"vslideup", 0b001110>;
defm VSLIDEDOWN : binary_arithmetic_integer_xu<"vslidedown", 0b001111>;

defm VADC : binary_arithmetic_integer_vxi_mask_in<"vadc", 0b010000,
                                                  vmask_all_lanes>;
defm VSBC : binary_arithmetic_integer_vx_mask_in<"vsbc", 0b010010,
                                                 vmask_all_lanes>;

// We could call it VMV but there is already VMV_S_X so that would be confusing.
defm VMV_V : unary_arithmetic_nomask_integer_vxi<"vmv.v", 0b010111>;
// The encoding of vmv.v and vmerge is the same except for vm.
// vmv.v "operates on all the elements" (vm=1)
// vmerge is encoded as if it operated "only in the true mask elements" (vm=0)
defm VMERGE : binary_arithmetic_integer_vxi_mask_in<"vmerge", 0b010111,
                                                    vmask_only_true>;

defm VMSEQ  : binary_arithmetic_integer_vxi<"vmseq",  0b011000>;
defm VMSNE  : binary_arithmetic_integer_vxi<"vmsne",  0b011001>;
defm VMSLTU : binary_arithmetic_integer_vx<"vmsltu",  0b011010>;
defm VMSLT  : binary_arithmetic_integer_vx<"vmslt",   0b011011>;
defm VMSLEU : binary_arithmetic_integer_vxu<"vmsleu", 0b011100>;
defm VMSLE  : binary_arithmetic_integer_vxi<"vmsle",  0b011101>;
defm VMSGTU : binary_arithmetic_integer_xu<"vmsgtu",  0b011110>;
defm VMSGT  : binary_arithmetic_integer_xi<"vmsgt",   0b011111>;

defm VSADDU : binary_arithmetic_integer_vxi<"vsaddu", 0b100000>;
defm VSADD  : binary_arithmetic_integer_vxi<"vsadd",  0b100001>;
defm VSSUBU : binary_arithmetic_integer_vx<"vssubu",  0b100010>;
defm VSSUB  : binary_arithmetic_integer_vx<"vssub",   0b100011>;
defm VAADD  : binary_arithmetic_integer_vxi<"vaadd",  0b100100>;
defm VSLL   : binary_arithmetic_integer_vxi<"vsll",   0b100101>;
defm VASUB  : binary_arithmetic_integer_vx<"vasub",  0b100110>;
defm VSMUL  : binary_arithmetic_integer_vx<"vsmul",  0b100111>;
defm VSRL   : binary_arithmetic_integer_vxi<"vsrl",   0b101000>;
defm VSRA   : binary_arithmetic_integer_vxi<"vsra",   0b101001>;
defm VSSRL  : binary_arithmetic_integer_vxi<"vssrl",  0b101010>;
defm VSSRA  : binary_arithmetic_integer_vxi<"vssra",  0b101011>;
defm VNSRL  : binary_arithmetic_integer_vxi<"vnsrl",  0b101100>;
defm VNSRA  : binary_arithmetic_integer_vxi<"vnsra",  0b101101>;
defm VNCLIPU : binary_arithmetic_integer_vxi<"vnclipu", 0b101110>;
defm VNCLIP  : binary_arithmetic_integer_vxi<"vnclip",  0b101111>;

defm VWREDSUMU : binary_arithmetic_integer_s<"vwredsumu", 0b110000>;
defm VWREDSUM  : binary_arithmetic_integer_s<"vwredsum",  0b110001>;

defm VDOTU : binary_arithmetic_integer_v<"vdotu", 0b111000>;
defm VDOT  : binary_arithmetic_integer_v<"vdot",  0b111001>;

defm VWSMACCU : binary_arithmetic_integer_vx<"vwsmaccu", 0b111100>;
defm VWSMACC  : binary_arithmetic_integer_vx<"vwsmacc",  0b111101>;
defm VWSMSACU : binary_arithmetic_integer_vx<"vwsmsacu", 0b111110>;
defm VWSMSAC  : binary_arithmetic_integer_vx<"vwsmsac",  0b111111>;

defm VREDSUM  : binary_arithmetic_mask_integer_s<"vredsum",  0b000000>;
defm VREDAND  : binary_arithmetic_mask_integer_s<"vredand",  0b000001>;
defm VREDOR   : binary_arithmetic_mask_integer_s<"vredor",   0b000010>;
defm VREDXOR  : binary_arithmetic_mask_integer_s<"vredxor",  0b000011>;
defm VREDMINU : binary_arithmetic_mask_integer_s<"vredminu", 0b000100>;
defm VREDMIN  : binary_arithmetic_mask_integer_s<"vredmin",  0b000101>;
defm VREDMAXU : binary_arithmetic_mask_integer_s<"vredmaxu", 0b000110>;
defm VREDMAX  : binary_arithmetic_mask_integer_s<"vredmax",  0b000111>;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE] in
def VEXT_X_V : RVInstEPIOPV_MVV<
    0b001100,
    (outs GPR:$rd),
    (ins EPIVR:$rs2, GPR:$rs1),
    "vext.x.v", "${rd}, ${rs2}, ${rs1}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE], rs2 = 0b00000 in
def VMV_S_X : RVInstEPIOPV_MVX<
    0b001101,
    (outs EPIVR:$rd),
    (ins GPR:$rs1),
    "vmv.s.x", "${rd}, ${rs1}">;

defm VSLIDE1UP   : binary_arithmetic_mask_integer_x<"vslide1up",   0b001110>;
defm VSLIDE1DOWN : binary_arithmetic_mask_integer_x<"vslide1down", 0b001111>;

defm VMPOPC    : unary_mask_integer_m<"vmpopc",  0b010100>;
defm VMFIRST   : unary_mask_integer_m<"vmfirst", 0b010101>;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE] in
def VCOMPRESS_VM : RVInstEPIOPV_MVV<
     0b010111,
     (outs EPIVR:$rd),
     (ins EPIVR:$rs2, EPIVR:$rs1),
     "vcompress.vm", "${rd}, ${rs2}, ${rs1}">;

defm VMANDNOT  : binary_mask<"vmandnot", 0b011000>;
defm VMAND     : binary_mask<"vmand",    0b011001>;
defm VMOR      : binary_mask<"vmor",     0b011010>;
defm VMXOR     : binary_mask<"vmxor",    0b011011>;
defm VMORNOT   : binary_mask<"vmornot",  0b011100>;
defm VMNAND    : binary_mask<"vmnand",   0b011101>;
defm VMNOR     : binary_mask<"vmnor",    0b011110>;
defm VMXNOR    : binary_mask<"vmxnor",   0b011111>;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE], rs1 = 0b00001 in
def VMSBF_M : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins EPIVR:$rs2),
  "vmsbf.m", "${rd}, ${rs2}">;
let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
    Uses = [VL, VTYPE], rs1 = 0b00001 in
def VMSBF_M_MASK : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins EPIVR:$rs2, vectormask:$vm),
  "vmsbf.m", "${rd}, ${rs2}, ${vm}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE], rs1 = 0b00010 in
def VMSOF_M : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins EPIVR:$rs2),
  "vmsof.m", "${rd}, ${rs2}">;
let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
    Uses = [VL, VTYPE], rs1 = 0b00010 in
def VMSOF_M_MASK : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins EPIVR:$rs2, vectormask:$vm),
  "vmsof.m", "${rd}, ${rs2}, ${vm}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE], rs1 = 0b00011 in
def VMSIF_M : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins EPIVR:$rs2),
  "vmsif.m", "${rd}, ${rs2}">;
let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
    Uses = [VL, VTYPE], rs1 = 0b00011 in
def VMSIF_M_MASK : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins EPIVR:$rs2, vectormask:$vm),
  "vmsif.m", "${rd}, ${rs2}, ${vm}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE], rs1 = 0b10000 in
def VIOTA_M : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins EPIVR:$rs2),
  "viota.m", "${rd}, ${rs2}">;
let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
    Uses = [VL, VTYPE], rs1 = 0b10000 in
def VIOTA_M_MASK : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins EPIVR:$rs2, vectormask:$vm),
  "viota.m", "${rd}, ${rs2}, ${vm}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE], rs1 = 0b10001, rs2 = 0b00000 in
def VID_V : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins),
  "vid.v", "${rd}">;
let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
    Uses = [VL, VTYPE], rs1 = 0b10001, rs2 = 0b00000 in
def VID_V_MASK : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins vectormask:$vm),
  "vid.v", "${rd}, ${vm}">;

defm VDIVU   : binary_arithmetic_mask_integer_vx<"vdivu",   0b100000>;
defm VDIV    : binary_arithmetic_mask_integer_vx<"vdiv",    0b100001>;
defm VREMU   : binary_arithmetic_mask_integer_vx<"vremu",   0b100010>;
defm VREM    : binary_arithmetic_mask_integer_vx<"vrem",    0b100011>;
defm VMULHU  : binary_arithmetic_mask_integer_vx<"vmulhu",  0b100100>;
defm VMUL    : binary_arithmetic_mask_integer_vx<"vmul",    0b100101>;
defm VMULHSU : binary_arithmetic_mask_integer_vx<"vmulhsu", 0b100110>;
defm VMULH   : binary_arithmetic_mask_integer_vx<"vmulh",   0b100111>;

defm VMADD   : binary_arithmetic_mask_integer_vx_destructive<"vmadd", 0b101001, "v">;
defm VMSUB   : binary_arithmetic_mask_integer_vx_destructive<"vmsub", 0b101011, "v">;
defm VMACC   : binary_arithmetic_mask_integer_vx_destructive<"vmacc", 0b101101, "v">;
defm VMSAC   : binary_arithmetic_mask_integer_vx_destructive<"vmsac", 0b101111, "v">;

defm VWADDU   : binary_arithmetic_mask_integer_vx<"vwaddu",  0b110000>;
defm VWADD    : binary_arithmetic_mask_integer_vx<"vwadd",   0b110001>;
defm VWSUBU   : binary_arithmetic_mask_integer_vx<"vwsubu",  0b110010>;
defm VWSUB    : binary_arithmetic_mask_integer_vx<"vwsub",   0b110011>;
defm VWADDU   : binary_arithmetic_mask_integer_wx<"vwaddu",  0b110100>;
defm VWADD    : binary_arithmetic_mask_integer_wx<"vwadd",   0b110101>;
defm VWSUBU   : binary_arithmetic_mask_integer_wx<"vwsubu",  0b110110>;
defm VWSUB    : binary_arithmetic_mask_integer_wx<"vwsub",   0b110111>;
defm VWMULU   : binary_arithmetic_mask_integer_vx<"vwmulu",  0b111000>;
defm VWMULSU  : binary_arithmetic_mask_integer_vx<"vwmulsu", 0b111010>;
defm VWMUL    : binary_arithmetic_mask_integer_vx<"vwmul",   0b111011>;

// These are wrong
defm VWMACCU  : binary_arithmetic_mask_integer_vx<"vwmaccu", 0b111100>;
defm VWMACC   : binary_arithmetic_mask_integer_vx<"vwmacc",  0b111101>;
defm VWMSACU  : binary_arithmetic_mask_integer_vx<"vwmsacu", 0b111110>;
defm VWMSAC   : binary_arithmetic_mask_integer_vx<"vwmsac",  0b111111>;

defm VFADD     : binary_arithmetic_float_vf<"vfadd",    0b000000>;
defm VFREDSUM  : binary_arithmetic_float_vs<"vfredsum",  0b000001>;
defm VFSUB     : binary_arithmetic_float_vf<"vfsub",    0b000010>;
defm VFREDOSUM : binary_arithmetic_float_vs<"vfredosum", 0b000011>;
defm VFMIN     : binary_arithmetic_float_vf<"vfmin",    0b000100>;
defm VFREDMIN  : binary_arithmetic_float_vs<"vfredmin",  0b000101>;
defm VFMAX     : binary_arithmetic_float_vf<"vfmax",    0b000110>;
defm VFREDMAX  : binary_arithmetic_float_vs<"vfredmax",  0b000111>;
defm VFSGNJ    : binary_arithmetic_float_vf<"vfsgnj",   0b001000>;
defm VFSGNJN   : binary_arithmetic_float_vf<"vfsgnjn",   0b001001>;
defm VFSGNJX   : binary_arithmetic_float_vf<"vfsgnjx",   0b001010>;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE], rs1 = 0b00000 in
def VFMV_F_S : RVInstEPIOPV_FVV<0b001100,
           (outs FPR64:$rd),
           (ins EPIVR:$rs2),
           "vfmv.f.s", "${rd}, ${rs2}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE], rs2 = 0b00000 in
def VFMV_S_F : RVInstEPIOPV_FVF<0b001101,
           (outs EPIVR:$rd),
           (ins FPR64:$rs1),
           "vfmv.s.f", "${rd}, ${rs1}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    rs2 = 0b00000, Uses = [VL, VTYPE] in
def VFMV_V_F : RVInstEPIOPV_FVF<0b010111,
           (outs EPIVR:$rd),
           (ins FPR64:$rs1),
           "vfmv.v.f",
           "${rd}, ${rs1}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_only_true.Value,
    Uses = [VL, VTYPE] in
def VFMERGE_VFM : RVInstEPIOPV_FVF<0b010111,
           (outs EPIVR:$rd),
           (ins EPIVR:$rs2, FPR64:$rs1, EPIMaskVR:$mask),
           "vfmerge.vfm",
           "${rd}, ${rs2}, ${rs1}, v0">;

defm VMFEQ  : binary_arithmetic_float_vf<"vmfeq",  0b011000>;
defm VMFLE  : binary_arithmetic_float_vf<"vmfle",  0b011001>;
defm VMFORD : binary_arithmetic_float_vf<"vmford", 0b011010>;
defm VMFLT  : binary_arithmetic_float_vf<"vmflt",  0b011011>;
defm VMFNE  : binary_arithmetic_float_vf<"vmfne",  0b011100>;
defm VMFGT  : binary_arithmetic_float_f<"vmfgt",   0b011101>;
defm VMFGE  : binary_arithmetic_float_f<"vmfge",   0b011111>;

defm VFDIV  : binary_arithmetic_float_vf<"vfdiv",  0b100000>;
defm VFRDIV : binary_arithmetic_float_vf<"vfrdiv", 0b100001>;

defm VFCVT_XU_F : unary_arithmetic_float_v<"vfcvt.xu.f", 0b100010, 0b00000>;
defm VFCVT_X_F  : unary_arithmetic_float_v<"vfcvt.x.f",  0b100010, 0b00001>;
defm VFCVT_F_XU : unary_arithmetic_float_v<"vfcvt.f.xu", 0b100010, 0b00010>;
defm VFCVT_F_X  : unary_arithmetic_float_v<"vfcvt.f.x",  0b100010, 0b00011>;

defm VFWCVT_XU_F : unary_arithmetic_float_v<"vfwcvt.xu.f", 0b100010, 0b01000>;
defm VFWCVT_X_F  : unary_arithmetic_float_v<"vfwcvt.x.f",  0b100010, 0b01001>;
defm VFWCVT_F_XU : unary_arithmetic_float_v<"vfwcvt.f.xu", 0b100010, 0b01010>;
defm VFWCVT_F_X  : unary_arithmetic_float_v<"vfwcvt.f.x",  0b100010, 0b01011>;
defm VFWCVT_F_F  : unary_arithmetic_float_v<"vfwcvt.f.f",  0b100010, 0b01100>;

defm VFNCVT_XU_F : unary_arithmetic_float_v<"vfncvt.xu.f", 0b100010, 0b10000>;
defm VFNCVT_X_F  : unary_arithmetic_float_v<"vfncvt.x.f",  0b100010, 0b10001>;
defm VFNCVT_F_XU : unary_arithmetic_float_v<"vfncvt.f.xu", 0b100010, 0b10010>;
defm VFNCVT_F_X  : unary_arithmetic_float_v<"vfncvt.f.x",  0b100010, 0b10011>;
defm VFNCVT_F_F  : unary_arithmetic_float_v<"vfncvt.f.f",  0b100010, 0b10100>;

defm VFSQRT  : unary_arithmetic_float_v<"vfsqrt", 0b100011, 0b00000>;
defm VFCLASS : unary_arithmetic_float_v<"vfclass", 0b100011, 0b10000>;

defm VFMUL   : binary_arithmetic_float_vf<"vfmul", 0b100100>;

defm VFMADD  : binary_arithmetic_float_vf_destructive<"vfmadd",  0b101000>;
defm VFNMADD : binary_arithmetic_float_vf_destructive<"vfnmadd", 0b101001>;
defm VFMSUB  : binary_arithmetic_float_vf_destructive<"vfmsub",  0b101010>;
defm VFNMSUB : binary_arithmetic_float_vf_destructive<"vfnmsub", 0b101011>;
defm VFMACC  : binary_arithmetic_float_vf_destructive<"vfmacc",  0b101100>;
defm VFNMACC : binary_arithmetic_float_vf_destructive<"vfnmacc", 0b101101>;
defm VFMSAC  : binary_arithmetic_float_vf_destructive<"vfmsac",  0b101110>;
defm VFNMSAC : binary_arithmetic_float_vf_destructive<"vfnmsac", 0b101111>;

defm VFWADD     : binary_arithmetic_float_vf<"vfwadd",    0b110000>;
defm VFWREDSUM  : binary_arithmetic_float_vs<"vfwredsum",  0b110001>;
defm VFWSUB     : binary_arithmetic_float_vf<"vfwsub",    0b110010>;
defm VFWREDOSUM : binary_arithmetic_float_vs<"vfwredosum", 0b110011>;
defm VFWADD     : binary_arithmetic_float_wf<"vfwadd",    0b110100>;
defm VFWSUB     : binary_arithmetic_float_wf<"vfwsub",    0b110110>;

defm VFWMUL     : binary_arithmetic_float_vf<"vfwmul",    0b111000>;
defm VFDOT      : binary_arithmetic_float_vf<"vfdot",     0b111001>;

defm VFWMACC  : binary_arithmetic_float_vf_destructive<"vfwmacc",  0b111100>;
defm VFWNMACC : binary_arithmetic_float_vf_destructive<"vfwnmacc", 0b111101>;
defm VFWMSAC  : binary_arithmetic_float_vf_destructive<"vfwmsac",  0b111110>;
defm VFWNMSAC : binary_arithmetic_float_vf_destructive<"vfwnmsac", 0b111111>;

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Defs = [VL, VTYPE] in
def VSETVLI : RVInst<
  (outs GPR:$rd),
  (ins GPR:$rs1, vectorelementwidth:$vsew, vectormultiplier:$vlmul),
  "vsetvli", "${rd}, ${rs1}, ${vsew}, ${vlmul}", [],
  InstFormatOther>
{
  bits<5> rd;
  bits<5> rs1;
  bits<10> imm10;
  bits<3> vsew;
  bits<2> vlmul;

  let Inst{31} = 0b0;
  let Inst{30-25} = 0b000000;
  let Inst{24-22} = vsew;
  let Inst{21-20} = vlmul;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b111;
  let Inst{11-7} = rd;
  let Opcode = OPC_V.Value;
}

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Defs = [VL, VTYPE] in
def VSETVL : RVInst<
  (outs GPR:$rd),
  (ins GPR:$rs1, GPR:$rs2),
  "vsetvl", "${rd}, ${rs1}, ${rs2}", [],
  InstFormatOther>
{
  bits<5> rd;
  bits<5> rs1;
  bits<5> rs2;

  let Inst{31} = 0b1;
  let Inst{30-25} = 0b000000;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b111;
  let Inst{11-7} = rd;
  let Opcode = OPC_V.Value;
}


let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 0,
    Uses = [VL] in
def PseudoReadVL : Pseudo<(outs GPR:$rd),
                          (ins), [], "rdvl $rd">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1,
     usesCustomInserter = 1 in
def PseudoVSCALE : Pseudo<(outs GPR:$rd), (ins), [], "">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 0,
    Uses = [VTYPE] in
def PseudoReadVTYPE : Pseudo<(outs GPR:$rd),
                          (ins), [], "rdvtype $rd">;

// TODO: Unit-stride fault-only first loads
defm VS : vector_store<"vs">;
defm VL : vector_load<"vl">;

//===----------------------------------------------------------------------===//
// Pseudo instructions we need for SPILL and RELOAD
//===----------------------------------------------------------------------===//
let hasSideEffects = 1, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1,
    Uses = [VL, VTYPE] in
def PseudoVSPILL : Pseudo<(outs), (ins EPIVR:$rs1, GPR:$rs2), [], "">;

let hasSideEffects = 1, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1,
    Uses = [VL, VTYPE] in
def PseudoVRELOAD : Pseudo<(outs EPIVR:$rs1), (ins GPR:$rs2), [], "">;


//===----------------------------------------------------------------------===//
// End of Instructions
//===----------------------------------------------------------------------===//
}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

class VectorTypeInfo<ValueType Vec, ValueType Mas, int Sew, EPIVReg Reg>
{
  ValueType Vector = Vec;
  ValueType Mask = Mas;
  int SEW = Sew;
  EPIVReg RegClass = Reg;
}

class FloatVectorTypeInfo<ValueType Vec, ValueType Mas, int Sew, EPIVReg Reg,
                          ValueType Scal, RegisterClass ScalarReg>
    : VectorTypeInfo<Vec, Mas, Sew, Reg>
{
  ValueType Scalar = Scal;
  RegisterClass ScalarRegClass = ScalarReg;
}

class VectorTypeInfoToWide<VectorTypeInfo vti, VectorTypeInfo wti>
{
  VectorTypeInfo Vti = vti;
  VectorTypeInfo Wti = wti;
}

class FloatVectorTypeInfoToWide<FloatVectorTypeInfo fvti, FloatVectorTypeInfo fwti>
{
  FloatVectorTypeInfo FVti = fvti;
  FloatVectorTypeInfo FWti = fwti;
}

defset list<VectorTypeInfo> AllVectors = {

  defset list<VectorTypeInfo> AllIntegerVectors = {
    defset list<VectorTypeInfo> NoGroupIntegerVectors = {
//    def Vtype1xi8  : VectorTypeInfo<nxv1i8,  nxv1i1, 8,  EPIVR>; // FIXME illegal type
//    def Vtype2xi8  : VectorTypeInfo<nxv2i8,  nxv2i1, 8,  EPIVR>; // FIXME illegal type
//    def Vtype4xi8  : VectorTypeInfo<nxv4i8,  nxv4i1, 8,  EPIVR>; // FIXME illegal type
      def Vtype8xi8  : VectorTypeInfo<nxv8i8,  nxv8i1, 8,  EPIVR>;

//    def Vtype1xi16 : VectorTypeInfo<nxv1i16, nxv1i1, 16, EPIVR>; // FIXME illegal type
//    def Vtype2xi16 : VectorTypeInfo<nxv2i16, nxv2i1, 16, EPIVR>; // FIXME illegal type
      def Vtype4xi16 : VectorTypeInfo<nxv4i16, nxv4i1, 16, EPIVR>;

//    def Vtype1xi32 : VectorTypeInfo<nxv1i32, nxv1i1, 32, EPIVR>; // FIXME illegal type
      def Vtype2xi32 : VectorTypeInfo<nxv2i32, nxv2i1, 32, EPIVR>;

      def Vtype1xi64 : VectorTypeInfo<nxv1i64, nxv1i1, 64, EPIVR>;
    }

    defset list<VectorTypeInfo> GroupIntegerVectors = {
      def Vtype16xi8  : VectorTypeInfo<nxv16i8,  nxv16i1, 8,  EPIVR2>;
      def Vtype32xi8  : VectorTypeInfo<nxv32i8,  nxv32i1, 8,  EPIVR4>;
//    def Vtype64xi8  : VectorTypeInfo<nxv64i8,  nxv64i1, 8,  EPIVR8>; // FIXME undefined type

      def Vtype8xi16  : VectorTypeInfo<nxv8i16,  nxv8i1,  16, EPIVR2>;
      def Vtype16xi16 : VectorTypeInfo<nxv16i16, nxv16i1, 16, EPIVR4>;
      def Vtype32xi16 : VectorTypeInfo<nxv32i16, nxv32i1, 16, EPIVR8>;

      def Vtype4xi32  : VectorTypeInfo<nxv4i32,  nxv4i1,  32, EPIVR2>;
      def Vtype8xi32  : VectorTypeInfo<nxv8i32,  nxv8i1,  32, EPIVR4>;
      def Vtype16xi32 : VectorTypeInfo<nxv16i32, nxv16i1, 32, EPIVR8>;

      def Vtype2xi64  : VectorTypeInfo<nxv2i64,  nxv2i1,  64, EPIVR2>;
      def Vtype4xi64  : VectorTypeInfo<nxv4i64,  nxv4i1,  64, EPIVR4>;
      def Vtype8xi64  : VectorTypeInfo<nxv8i64,  nxv8i1,  64, EPIVR8>;
    }
  }

  defset list<FloatVectorTypeInfo> AllFloatVectors = {
    defset list<FloatVectorTypeInfo> NoGroupFloatVectors = {
//    def Vtype1xf32 : FloatVectorTypeInfo<nxv1f32, nxv1i1, 32, EPIVR, f32, FPR32>; // FIXME illegal type
      def Vtype2xf32 : FloatVectorTypeInfo<nxv2f32, nxv2i1, 32, EPIVR, f32, FPR32>;

      def Vtype1xf64 : FloatVectorTypeInfo<nxv1f64, nxv1i1, 64, EPIVR, f64, FPR64>;
    }

    def Vtype4xf32  : FloatVectorTypeInfo<nxv4f32,  nxv4i1,  32, EPIVR2, f32, FPR32>;
    def Vtype8xf32  : FloatVectorTypeInfo<nxv8f32,  nxv8i1,  32, EPIVR4, f32, FPR32>;
    def Vtype16xf32 : FloatVectorTypeInfo<nxv16f32, nxv16i1, 32, EPIVR8, f32, FPR32>;

    def Vtype2xf64  : FloatVectorTypeInfo<nxv2f64,  nxv2i1,  64, EPIVR2, f64, FPR64>;
    def Vtype4xf64  : FloatVectorTypeInfo<nxv4f64,  nxv4i1,  64, EPIVR4, f64, FPR64>;
    def Vtype8xf64  : FloatVectorTypeInfo<nxv8f64,  nxv8i1,  64, EPIVR8, f64, FPR64>;
  }
}

// This functor is used to obtain the int vector type that has the same SEW and
// multiplier as the input parameter type
class GetIntVectorTypeInfo<VectorTypeInfo vti>
{
  // Equivalent integer vector type. Eg.
  //   Vtype8xi8 → Vtype8xi8 (identity)
  //   Vtype4xf64 → Vtype4xi64
  VectorTypeInfo Vti =
    !cast<VectorTypeInfo>(
      !subst("f", "i", !cast<string>(vti))
    );
}

// This functor is used to obtain the float vector type that has the same SEW
// and multiplier as the input parameter type
class GetFloatVectorTypeInfo<VectorTypeInfo vti>
{
  // Equivalent float vector type. Eg.
  //   Vtype32xi16 → Vtype32xf16
  //   Vtype2xf32 → Vtype2xf32 (identity)
  VectorTypeInfo FVti =
    !cast<VectorTypeInfo>(
      !subst("i", "f", !cast<string>(vti))
    );
}

defset list<VectorTypeInfoToWide> AllWideableIntVectors = {
//def : VectorTypeInfoToWide<Vtype1xi8,   Vtype1xi16>; // FIXME illegal type
//def : VectorTypeInfoToWide<Vtype2xi8,   Vtype2xi16>; // FIXME illegal type
//def : VectorTypeInfoToWide<Vtype4xi8,   Vtype4xi16>; // FIXME illegal type
  def : VectorTypeInfoToWide<Vtype8xi8,   Vtype8xi16>;
  def : VectorTypeInfoToWide<Vtype16xi8,  Vtype16xi16>;
  def : VectorTypeInfoToWide<Vtype32xi8,  Vtype32xi16>;

//def : VectorTypeInfoToWide<Vtype1xi16,  Vtype1xi32>; // FIXME illegal type
//def : VectorTypeInfoToWide<Vtype2xi16,  Vtype2xi32>; // FIXME illegal type
  def : VectorTypeInfoToWide<Vtype4xi16,  Vtype4xi32>;
  def : VectorTypeInfoToWide<Vtype8xi16,  Vtype8xi32>;
  def : VectorTypeInfoToWide<Vtype16xi16, Vtype16xi32>;

//def : VectorTypeInfoToWide<Vtype1xi32,  Vtype1xi64>; // FIXME illegal type
  def : VectorTypeInfoToWide<Vtype2xi32,  Vtype2xi64>;
  def : VectorTypeInfoToWide<Vtype4xi32,  Vtype4xi64>;
  def : VectorTypeInfoToWide<Vtype8xi32,  Vtype8xi64>;
// FIXME what about these?
//def : VectorTypeInfoToWide<Vtype1xi64,  /* FIXME Vtype1xi128 */ i1>;
//def : VectorTypeInfoToWide<Vtype2xi64,  /* FIXME Vtype2xi128 */ i1>;
//def : VectorTypeInfoToWide<Vtype4xi64,  /* FIXME Vtype4xi128 */ i1>;
}

defset list<VectorTypeInfoToWide> AllWideableIntToFloatVectors = {
//def : VectorTypeInfoToWide<Vtype1xi16,  Vtype1xi32>; // FIXME illegal type
//def : VectorTypeInfoToWide<Vtype2xi16,  Vtype2xi32>; // FIXME illegal type
  def : VectorTypeInfoToWide<Vtype4xi16,  Vtype4xi32>;
  def : VectorTypeInfoToWide<Vtype8xi16,  Vtype8xi32>;
  def : VectorTypeInfoToWide<Vtype16xi16, Vtype16xi32>;

//def : VectorTypeInfoToWide<Vtype1xi32,  Vtype1xi64>; // FIXME illegal type
  def : VectorTypeInfoToWide<Vtype2xi32,  Vtype2xi64>;
  def : VectorTypeInfoToWide<Vtype4xi32,  Vtype4xi64>;
  def : VectorTypeInfoToWide<Vtype8xi32,  Vtype8xi64>;
}

defset list<FloatVectorTypeInfoToWide> AllWideableFloatVectors = {
//def : FloatVectorTypeInfoToWide<Vtype1xf32, Vtype1xf64>; // FIXME illegal type
  def : FloatVectorTypeInfoToWide<Vtype2xf32, Vtype2xf64>;
  def : FloatVectorTypeInfoToWide<Vtype4xf32, Vtype4xf64>;
  def : FloatVectorTypeInfoToWide<Vtype8xf32, Vtype8xf64>;
// FIXME what about these?
//def : FloatVectorTypeInfoToWide<Vtype1xf64, /* FIXME: Vtype1xf128 */ i1>;
//def : FloatVectorTypeInfoToWide<Vtype2xf64, /* FIXME: Vtype2xf128 */ i1>;
//def : FloatVectorTypeInfoToWide<Vtype4xf64, /* FIXME: Vtype4xf128 */ i1>;
}

class MaskTypeInfo<ValueType Mas, int Sew, int Vlmul, ValueType Ivt> {
  ValueType Mask = Mas;
  // {SEW, VLMul} values set a valid VType to deal with this mask type.
  int SEW = Sew; // FIXME: computed as ELEN/vscale, to be used for loading/storing partial masks
  int VLMul = Vlmul; // Convention: Minimum VLMul where this mask type appears.
  ValueType IntegerVector = Ivt;
}

defset list<MaskTypeInfo> AllMasks = {
  def : MaskTypeInfo<nxv1i1,  64, 1, nxv1i64>; // FIXME SEW should be ELEN/1
  def : MaskTypeInfo<nxv2i1,  32, 1, nxv2i32>; // FIXME SEW should be ELEN/2
  def : MaskTypeInfo<nxv4i1,  16, 1, nxv4i16>; // FIXME SEW should be ELEN/4
  def : MaskTypeInfo<nxv8i1,  8,  1, nxv8i8>;  // FIXME SEW should be ELEN/8
  def : MaskTypeInfo<nxv16i1, 8,  2, nxv16i4>;  // FIXME SEW should be ELEN/16, SEW < 8
  def : MaskTypeInfo<nxv32i1, 8,  4, nxv32i2>;  // FIXME SEW should be ELEN/32, SEW < 8
//def : MaskTypeInfo<nxv64i1, 8,  8, nxv64i1>;  // FIXME SEW should be ELEN/64, SEW < 8, undefined type
}

class EPIVRegToWide<EPIVReg Reg, EPIVReg WideReg> {
  EPIVReg RegClass = Reg;
  EPIVReg WideRegClass = WideReg;
}

defset list<EPIVRegToWide> AllEPIVRegToWidePairs = {
  def : EPIVRegToWide<EPIVR, EPIVR2>;
  def : EPIVRegToWide<EPIVR2, EPIVR4>;
  def : EPIVRegToWide<EPIVR4, EPIVR8>;
}

def EPIIntrClassID : GenericEnum {
  let FilterClass = "EPIIntrinsicClassID";
}

def EPIIntrinsicsTable : GenericTable {
  let FilterClass = "EPIIntrinsic";
  let CppTypeName = "EPIIntrinsicInfo";
  let Fields = [ "IntrinsicID", "ClassID", "ExtendOperand", "MaskOperand",
                 "GVLOperand" ];
  let PrimaryKey = [ "IntrinsicID" ];
  let PrimaryKeyName = "getEPIIntrinsicInfo";

  GenericEnum TypeOf_ClassID = EPIIntrClassID;
}

class EPIPseudo {
  Pseudo Pseudo = !cast<Pseudo>(NAME);
  Instruction BaseInstr;
  bits<8> VLIndex;
  bits<8> SEWIndex;
  bits<8> MergeOpIndex;
  bits<8> VLMul;
}

def EPIPseudosTable : GenericTable {
  let FilterClass = "EPIPseudo";
  let CppTypeName = "EPIPseudoInfo";
  let Fields = [ "Pseudo", "BaseInstr", "VLIndex", "SEWIndex", "MergeOpIndex", "VLMul" ];
  let PrimaryKey = [ "Pseudo" ];
  let PrimaryKeyName = "getEPIPseudoInfo";
}


multiclass pseudo_nullary<EPIVReg result_reg_class,
                          int vlmul,
                          string constraints = ""> {
  let Uses = [VL, VTYPE],
      VLIndex = 1, SEWIndex = 2, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;

  let Constraints = Join<[constraints, "$rd = $merge"], ",">.ret,
      Uses = [VL, VTYPE], VLIndex = 3, SEWIndex = 4, MergeOpIndex = 1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)#"_MASK") in
    def "_MASK_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                                 (ins result_reg_class:$merge,
                                      vectormask:$vm,
                                      GPR:$vl, ixlenimm:$sew),
                                 []>,
                          EPIPseudo;
}

multiclass pseudo_nullary_v {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _V : pseudo_nullary<evr, evr.VLMul>;
    }
  }
}

multiclass pseudo_unary_nomask<EPIVReg result_reg_class,
                               EPIVReg op_reg_class,
                               int vlmul> {
  let Uses = [VL, VTYPE],
      VLIndex = 2, SEWIndex = 3, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins op_reg_class:$rs2, GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

// Special case for masking that does not have a merge operand.
multiclass pseudo_unary_mask_nomerge<EPIVReg result_reg_class,
                                     EPIVReg op_reg_class,
                                     int vlmul> {
  let Uses = [VL, VTYPE], VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)#"_MASK") in
    def "_MASK_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                                 (ins op_reg_class:$rs2, vectormask:$vm,
                                      GPR:$vl, ixlenimm:$sew),
                                 []>,
                          EPIPseudo;
}

multiclass pseudo_unary<EPIVReg result_reg_class,
                        EPIVReg op_reg_class,
                        int vlmul,
                        string constraints = ""> {

  let Constraints = constraints in
  defm "" : pseudo_unary_nomask<result_reg_class, op_reg_class, vlmul>;

  let Constraints = Join<[constraints, "$rd = $merge"], ",">.ret,
      Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = 1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)#"_MASK") in
    def "_MASK_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                                 (ins result_reg_class:$merge,
                                      op_reg_class:$rs2, vectormask:$vm,
                                      GPR:$vl, ixlenimm:$sew),
                                 []>,
                          EPIPseudo;
}

multiclass pseudo_unary_v_v {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _V : pseudo_unary<evr, evr, evr.VLMul>;
    }
  }
}

multiclass pseudo_unary_v_v_x_i_nomask
{
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _V : pseudo_unary_nomask<evr, evr, evr.VLMul>;
      defm _X : pseudo_unary_nomask<evr, GPR, evr.VLMul>;
      defm _I : pseudo_unary_nomask<evr, simm5, evr.VLMul>; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_unary_v_f_nomask {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _F : pseudo_unary_nomask<evr, FPR64, evr.VLMul>;
    }
  }
}

multiclass pseudo_unary_w_v {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegToWidePairs in
  {
    let VLMul = evr.RegClass.VLMul in
    {
      defm _V : pseudo_unary<evr.WideRegClass, evr.RegClass,
                             evr.RegClass.VLMul, "@earlyclobber $rd">;
    }
  }
}

multiclass pseudo_unary_v_w {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegToWidePairs in
  {
    let VLMul = evr.RegClass.VLMul in
    {
      defm _V : pseudo_unary<evr.RegClass, evr.WideRegClass,
                             evr.RegClass.VLMul, "@earlyclobber $rd">;
    }
  }
}

multiclass pseudo_binary_nomask<EPIVReg result_reg_class,
                                EPIVReg op1_reg_class,
                                RegisterClass op2_reg_class, // FIXME dagoperand
                                int vlmul, string constraints = ""> {
  let Constraints = constraints, Uses = [VL, VTYPE],
      VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins op1_reg_class:$rs2, op2_reg_class:$rs1,
                                 GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

multiclass pseudo_binary<EPIVReg result_reg_class,
                         EPIVReg op1_reg_class,
                         RegisterClass op2_reg_class, // FIXME dagoperand
                         int vlmul, string constraints = ""> {
  defm "" : pseudo_binary_nomask<result_reg_class, op1_reg_class, op2_reg_class,
                                 vlmul, constraints>;

  let Constraints = Join<[constraints, "$rd = $merge"], ",">.ret,
      Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)#"_MASK") in
    def "_MASK_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                                 (ins result_reg_class:$merge,
                                      op1_reg_class:$rs2, op2_reg_class:$rs1,
                                      vectormask:$vm, GPR:$vl, ixlenimm:$sew),
                                 []>,
                          EPIPseudo;
}

multiclass pseudo_binary_mask_in<EPIVReg result_reg_class,
                                 EPIVReg op1_reg_class,
                                 RegisterClass op2_reg_class,
                                 int vlmul> {
  let Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins op1_reg_class:$rs2, op2_reg_class:$rs1,
                                 EPIMaskVR:$maskop, GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

multiclass pseudo_binary_v_vv {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<evr, evr, evr, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_v_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VX : pseudo_binary<evr, evr, GPR, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_v_vs {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VS : pseudo_binary<evr, evr, evr, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_v_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<evr, evr, evr, evr.VLMul>;
      defm _VX : pseudo_binary<evr, evr, GPR, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_v_vx_vi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VX : pseudo_binary<evr, evr, GPR, evr.VLMul>;
      defm _VI : pseudo_binary<evr, evr, simm5, evr.VLMul>; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_v_vx_vu {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VX : pseudo_binary<evr, evr, GPR, evr.VLMul>;
      defm _VU : pseudo_binary<evr, evr, uimm5, evr.VLMul>; // FIXME how can this work? uimm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_v_vv_vx_vi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<evr, evr, evr, evr.VLMul>;
      defm _VX : pseudo_binary<evr, evr, GPR, evr.VLMul>;
      defm _VI : pseudo_binary<evr, evr, simm5, evr.VLMul>; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

// FIXME:  VV, VX and VI are used here instead of WV, WX and WI because of how
// binary narrowing instructions are named in the spec. We have enquired them
// about this issue (https://github.com/riscv/riscv-v-spec/issues/246) but we
// haven't received any response yet
multiclass pseudo_binary_v_wv_wx_wi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegToWidePairs in
  {
    let VLMul = evr.RegClass.VLMul in
    {
      defm _VV : pseudo_binary<evr.RegClass, evr.WideRegClass, evr.RegClass,
                               evr.RegClass.VLMul, "@earlyclobber $rd">;
      defm _VX : pseudo_binary<evr.RegClass, evr.WideRegClass, GPR,
                               evr.RegClass.VLMul, "@earlyclobber $rd">;
      defm _VI : pseudo_binary<evr.RegClass, evr.WideRegClass, simm5,
                               evr.RegClass.VLMul, "@earlyclobber $rd">; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_v_vvm_vxm_vim {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VVM : pseudo_binary_mask_in<evr, evr, evr, evr.VLMul>;
      defm _VXM : pseudo_binary_mask_in<evr, evr, GPR, evr.VLMul>;
      defm _VIM : pseudo_binary_mask_in<evr, evr, simm5, evr.VLMul>; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_w_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegToWidePairs in
  {
    let VLMul = evr.RegClass.VLMul in
    {
      defm _VV : pseudo_binary<evr.WideRegClass, evr.RegClass, evr.RegClass,
                               evr.RegClass.VLMul, "@earlyclobber $rd">;
      defm _VX : pseudo_binary<evr.WideRegClass, evr.RegClass, GPR,
                               evr.RegClass.VLMul, "@earlyclobber $rd">;
    }
  }
}

multiclass pseudo_binary_m_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<EPIVR, evr, evr, evr.VLMul>;
      defm _VX : pseudo_binary<EPIVR, evr, GPR, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_m_vx_vi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VX : pseudo_binary<EPIVR, evr, GPR, evr.VLMul>;
      defm _VI : pseudo_binary<EPIVR, evr, simm5, evr.VLMul>; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_m_vx_vu {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VX : pseudo_binary<EPIVR, evr, GPR, evr.VLMul>;
      defm _VU : pseudo_binary<EPIVR, evr, uimm5, evr.VLMul>; // FIXME how can this work? uimm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_m_vv_vx_vi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<EPIVR, evr, evr, evr.VLMul>;
      defm _VX : pseudo_binary<EPIVR, evr, GPR, evr.VLMul>;
      defm _VI : pseudo_binary<EPIVR, evr, simm5, evr.VLMul>; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_m_vv_vx_vu {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<EPIVR, evr, evr, evr.VLMul>;
      defm _VX : pseudo_binary<EPIVR, evr, GPR, evr.VLMul>;
      defm _VU : pseudo_binary<EPIVR, evr, uimm5, evr.VLMul>; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_v_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VF : pseudo_binary<evr, evr, FPR64, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_v_vfm {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VFM : pseudo_binary_mask_in<evr, evr, FPR64, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_v_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<evr, evr, evr, evr.VLMul>;
      defm _VF : pseudo_binary<evr, evr, FPR64, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_w_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegToWidePairs in
  {
    let VLMul = evr.RegClass.VLMul in
    {
      defm _VV : pseudo_binary<evr.WideRegClass, evr.RegClass, evr.RegClass,
                               evr.RegClass.VLMul, "@earlyclobber $rd">;
      defm _VF : pseudo_binary<evr.WideRegClass, evr.RegClass, FPR64,
                               evr.RegClass.VLMul, "@earlyclobber $rd">;
    }
  }
}

multiclass pseudo_binary_m_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VF : pseudo_binary<EPIVR, evr, FPR64, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_m_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<EPIVR, evr, evr, evr.VLMul>;
      defm _VF : pseudo_binary<EPIVR, evr, FPR64, evr.VLMul>;
    }
  }
}

multiclass pseudo_ternary<EPIVReg result_reg_class,
                          EPIVReg op1_reg_class,
                          RegisterClass op2_reg_class,
                          int vlmul> { // FIXME dagoperand?
  let Constraints = "$rd = $rs3",
      Uses = [VL, VTYPE],
      VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins result_reg_class:$rs3, op1_reg_class:$rs1,
                                 op2_reg_class:$rs2, GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;

  let Constraints = "$rd = $rs3",
      Uses = [VL, VTYPE],
      VLIndex = 5, SEWIndex = 6, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)#"_MASK") in
    def "_MASK_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                                 (ins result_reg_class:$rs3, op1_reg_class:$rs1,
                                      op2_reg_class:$rs2, vectormask:$vm,
                                      GPR:$vl, ixlenimm:$sew),
                                 []>,
                          EPIPseudo;
}

multiclass pseudo_ternary_v_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_ternary<evr, evr, evr, evr.VLMul>;
      defm _VX : pseudo_ternary<evr, GPR, evr, evr.VLMul>;
    }
  }
}

multiclass pseudo_ternary_v_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllEPIVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_ternary<evr, evr, evr, evr.VLMul>;
      defm _VF : pseudo_ternary<evr, FPR64, evr, evr.VLMul>;
    }
  }
}

multiclass pseudo_unary_x_v {
  // A different pseudo is defined for each VLMul value, but (ungrouped) EPIVR
  // register operands are used in all of them.
  foreach Vlmul = [1, 2, 4, 8] in
  {
    let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1,
        VLMul = Vlmul in
    {
      defm _M : pseudo_unary_nomask<GPR, EPIVR, Vlmul>;
      defm _M : pseudo_unary_mask_nomerge<GPR, EPIVR, Vlmul>;
    }
  }
}

multiclass pseudo_binary_m_mm {
  // A different pseudo is defined for each VLMul value, but (ungrouped) EPIVR
  // register operands are used in all of them.
  foreach Vlmul = [1, 2, 4, 8] in
  {
    let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1,
        VLMul = Vlmul in
    {
      defm _MM : pseudo_binary_nomask<EPIVR, EPIVR, EPIVR, Vlmul>;
    }
  }
}

let Predicates = [HasExtEPI] in {

defm PseudoVADD        : pseudo_binary_v_vv_vx_vi;
defm PseudoVSUB        : pseudo_binary_v_vv_vx;
defm PseudoVRSUB       : pseudo_binary_v_vx_vi;
defm PseudoVMINU       : pseudo_binary_v_vv_vx;
defm PseudoVMIN        : pseudo_binary_v_vv_vx;
defm PseudoVMAXU       : pseudo_binary_v_vv_vx;
defm PseudoVMAX        : pseudo_binary_v_vv_vx;
defm PseudoVAND        : pseudo_binary_v_vv_vx_vi;
defm PseudoVOR         : pseudo_binary_v_vv_vx_vi;
defm PseudoVXOR        : pseudo_binary_v_vv_vx_vi;

defm PseudoVRGATHER    : pseudo_binary_v_vv_vx_vi;
defm PseudoVSLIDEUP    : pseudo_binary_v_vx_vu;
defm PseudoVSLIDEDOWN  : pseudo_binary_v_vx_vu;

defm PseudoVMSEQ       : pseudo_binary_m_vv_vx_vi;
defm PseudoVMSNE       : pseudo_binary_m_vv_vx_vi;
defm PseudoVMSLTU      : pseudo_binary_m_vv_vx;
defm PseudoVMSLT       : pseudo_binary_m_vv_vx;
defm PseudoVMSLEU      : pseudo_binary_m_vv_vx_vu;
defm PseudoVMSLE       : pseudo_binary_m_vv_vx_vi;

defm PseudoVMSGTU      : pseudo_binary_m_vx_vu;
defm PseudoVMSGT       : pseudo_binary_m_vx_vi;

defm PseudoVSADDU      : pseudo_binary_v_vv_vx_vi;
defm PseudoVSADD       : pseudo_binary_v_vv_vx_vi;
defm PseudoVSSUBU      : pseudo_binary_v_vv_vx;
defm PseudoVSSUB       : pseudo_binary_v_vv_vx;
defm PseudoVAADD       : pseudo_binary_v_vv_vx_vi;
defm PseudoVSLL        : pseudo_binary_v_vv_vx_vi;
defm PseudoVASUB       : pseudo_binary_v_vv_vx;
defm PseudoVSMUL       : pseudo_binary_v_vv_vx;
defm PseudoVSRL        : pseudo_binary_v_vv_vx_vi;
defm PseudoVSRA        : pseudo_binary_v_vv_vx_vi;
defm PseudoVSSRL       : pseudo_binary_v_vv_vx_vi;
defm PseudoVSSRA       : pseudo_binary_v_vv_vx_vi;

defm PseudoVNSRL       : pseudo_binary_v_wv_wx_wi;

defm PseudoVMV_V       : pseudo_unary_v_v_x_i_nomask;
defm PseudoVMERGE      : pseudo_binary_v_vvm_vxm_vim;

defm PseudoVDOTU       : pseudo_binary_v_vv;
defm PseudoVDOT        : pseudo_binary_v_vv;

defm PseudoVREDSUM     : pseudo_binary_v_vs;
defm PseudoVREDAND     : pseudo_binary_v_vs;
defm PseudoVREDOR      : pseudo_binary_v_vs;
defm PseudoVREDXOR     : pseudo_binary_v_vs;
defm PseudoVREDMINU    : pseudo_binary_v_vs;
defm PseudoVREDMIN     : pseudo_binary_v_vs;
defm PseudoVREDMAXU    : pseudo_binary_v_vs;
defm PseudoVREDMAX     : pseudo_binary_v_vs;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1,
    Uses = [VL, VTYPE], VLMul = EPIVR.VLMul, MergeOpIndex = -1 in
{
  let VLIndex = -1, SEWIndex = 3, BaseInstr = VEXT_X_V in
    def PseudoVEXT_X_V : Pseudo<(outs GPR:$rd),
                                (ins EPIVR:$rs2, GPR:$rs1, ixlenimm:$sew),
                                []>,
                         EPIPseudo;
  let VLIndex = 2, SEWIndex = 3, BaseInstr = VMV_S_X in
    def PseudoVMV_S_X: Pseudo<(outs EPIVR:$rd),
                              (ins GPR:$rs2, GPR:$vl, ixlenimm:$sew),
                              []>,
                       EPIPseudo;
}

defm PseudoVSLIDE1UP   : pseudo_binary_v_vx;
defm PseudoVSLIDE1DOWN : pseudo_binary_v_vx;

defm PseudoVDIVU       : pseudo_binary_v_vv_vx;
defm PseudoVDIV        : pseudo_binary_v_vv_vx;
defm PseudoVREMU       : pseudo_binary_v_vv_vx;
defm PseudoVREM        : pseudo_binary_v_vv_vx;
defm PseudoVMULHU      : pseudo_binary_v_vv_vx;
defm PseudoVMUL        : pseudo_binary_v_vv_vx;
defm PseudoVMULHSU     : pseudo_binary_v_vv_vx;
defm PseudoVMULH       : pseudo_binary_v_vv_vx;

defm PseudoVMADD       : pseudo_ternary_v_vv_vx;
defm PseudoVMSUB       : pseudo_ternary_v_vv_vx;
defm PseudoVMACC       : pseudo_ternary_v_vv_vx;
defm PseudoVMSAC       : pseudo_ternary_v_vv_vx;

defm PseudoVWADDU      : pseudo_binary_w_vv_vx;
defm PseudoVWADD       : pseudo_binary_w_vv_vx;
defm PseudoVWSUBU      : pseudo_binary_w_vv_vx;
defm PseudoVWSUB       : pseudo_binary_w_vv_vx;

defm PseudoVWMULU      : pseudo_binary_w_vv_vx;
defm PseudoVWMULSU     : pseudo_binary_w_vv_vx;
defm PseudoVWMUL       : pseudo_binary_w_vv_vx;

defm PseudoVFADD       : pseudo_binary_v_vv_vf;
defm PseudoVFREDSUM    : pseudo_binary_v_vs;
defm PseudoVFSUB       : pseudo_binary_v_vv_vf;
defm PseudoVFREDOSUM   : pseudo_binary_v_vs;
defm PseudoVFMIN       : pseudo_binary_v_vv_vf;
defm PseudoVFREDMIN    : pseudo_binary_v_vs;
defm PseudoVFMAX       : pseudo_binary_v_vv_vf;
defm PseudoVFREDMAX    : pseudo_binary_v_vs;
defm PseudoVFSGNJ      : pseudo_binary_v_vv_vf;
defm PseudoVFSGNJN     : pseudo_binary_v_vv_vf;
defm PseudoVFSGNJX     : pseudo_binary_v_vv_vf;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1,
    Uses = [VL, VTYPE], VLMul = EPIVR.VLMul, MergeOpIndex = -1 in
{
  let VLIndex = 2, SEWIndex = 3, BaseInstr = VFMV_F_S in
    def PseudoVFMV_F_S : Pseudo<(outs FPR64:$rd),
                                (ins EPIVR:$rs2, GPR:$vl, ixlenimm:$sew),
                                []>,
                         EPIPseudo;
  let VLIndex = 2, SEWIndex = 3, BaseInstr = VFMV_S_F in
    def PseudoVFMV_S_F : Pseudo<(outs EPIVR:$rd),
                                (ins FPR64:$rs2, GPR:$vl, ixlenimm:$sew),
                                []>,
                         EPIPseudo;
}

defm PseudoVMFEQ       : pseudo_binary_m_vv_vf;
defm PseudoVMFLE       : pseudo_binary_m_vv_vf;
defm PseudoVMFORD      : pseudo_binary_m_vv_vf;
defm PseudoVMFLT       : pseudo_binary_m_vv_vf;
defm PseudoVMFNE       : pseudo_binary_m_vv_vf;

defm PseudoVMFGT       : pseudo_binary_m_vf;
defm PseudoVMFGE       : pseudo_binary_m_vf;

defm PseudoVFDIV       : pseudo_binary_v_vv_vf;
defm PseudoVFRDIV      : pseudo_binary_v_vf;
defm PseudoVFMUL       : pseudo_binary_v_vv_vf;


defm PseudoVFMV_V      : pseudo_unary_v_f_nomask;
defm PseudoVFMERGE     : pseudo_binary_v_vfm;

defm PseudoVFMADD      : pseudo_ternary_v_vv_vf;
defm PseudoVFMSUB      : pseudo_ternary_v_vv_vf;
defm PseudoVFMACC      : pseudo_ternary_v_vv_vf;
defm PseudoVFMSAC      : pseudo_ternary_v_vv_vf;

defm PseudoVFNMADD     : pseudo_ternary_v_vv_vf;
defm PseudoVFNMSUB     : pseudo_ternary_v_vv_vf;
defm PseudoVFNMACC     : pseudo_ternary_v_vv_vf;
defm PseudoVFNMSAC     : pseudo_ternary_v_vv_vf;

defm PseudoVFWADD      : pseudo_binary_w_vv_vf;
defm PseudoVFWSUB      : pseudo_binary_w_vv_vf;
defm PseudoVFWMUL      : pseudo_binary_w_vv_vf;
defm PseudoVFDOT       : pseudo_binary_v_vv;

defm PseudoVFSQRT      : pseudo_unary_v_v;

defm PseudoVFCVT_XU_F  : pseudo_unary_v_v;
defm PseudoVFCVT_X_F   : pseudo_unary_v_v;
defm PseudoVFCVT_F_XU  : pseudo_unary_v_v;
defm PseudoVFCVT_F_X   : pseudo_unary_v_v;

defm PseudoVFWCVT_XU_F : pseudo_unary_w_v;
defm PseudoVFWCVT_X_F  : pseudo_unary_w_v;
defm PseudoVFWCVT_F_XU : pseudo_unary_w_v;
defm PseudoVFWCVT_F_X  : pseudo_unary_w_v;
defm PseudoVFWCVT_F_F  : pseudo_unary_w_v;

defm PseudoVFNCVT_XU_F : pseudo_unary_v_w;
defm PseudoVFNCVT_X_F  : pseudo_unary_v_w;
defm PseudoVFNCVT_F_XU : pseudo_unary_v_w;
defm PseudoVFNCVT_F_X  : pseudo_unary_v_w;
defm PseudoVFNCVT_F_F  : pseudo_unary_v_w;

defm PseudoVID         : pseudo_nullary_v;

defm PseudoVMPOPC      : pseudo_unary_x_v;
defm PseudoVMFIRST     : pseudo_unary_x_v;

defm PseudoVMANDNOT    : pseudo_binary_m_mm;
defm PseudoVMAND       : pseudo_binary_m_mm;
defm PseudoVMOR        : pseudo_binary_m_mm;
defm PseudoVMXOR       : pseudo_binary_m_mm;
defm PseudoVMORNOT     : pseudo_binary_m_mm;
defm PseudoVMNAND      : pseudo_binary_m_mm;
defm PseudoVMNOR       : pseudo_binary_m_mm;
defm PseudoVMXNOR      : pseudo_binary_m_mm;

// Load/store pseudo instructions

foreach evr = AllEPIVRegs in
foreach vlmul = [evr.VLMul] in
{
  let mayLoad = 1, mayStore = 0, hasSideEffects = 0,
      usesCustomInserter = 1,
      VLMul = vlmul in
  {
    let Uses = [VL, VTYPE],
        VLIndex = 2, SEWIndex = 3, MergeOpIndex = -1,
        BaseInstr = VLE_V in
      def PseudoVLE_V_M#vlmul
          : Pseudo<(outs evr:$rd),
                   (ins GPR:$rs1, GPR:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Constraints = "$rd = $merge", Uses = [VL, VTYPE],
        VLIndex = 4, SEWIndex = 5, MergeOpIndex = 1,
        BaseInstr = VLE_V_MASK in
      def PseudoVLE_V_MASK_M#vlmul
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, vectormask:$mask, GPR:$vl,
                    ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Uses = [VL, VTYPE],
        VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
        BaseInstr = VLSE_V in
      def PseudoVLSE_V_M#vlmul
          : Pseudo<(outs evr:$rd),
                   (ins GPR:$rs1, GPR:$rs2, GPR:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Constraints = "$rd = $merge", Uses = [VL, VTYPE],
        VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
        BaseInstr = VLSE_V_MASK in
      def PseudoVLSE_V_MASK_M#vlmul
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, GPR:$rs2, vectormask:$mask,
                    GPR:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Uses = [VL, VTYPE],
        VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
        BaseInstr = VLXE_V in
      def PseudoVLXE_V_M#vlmul
          : Pseudo<(outs evr:$rd),
                   (ins GPR:$rs1, evr:$rs2, GPR:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Constraints = "$rd = $merge", Uses = [VL, VTYPE],
        VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
        BaseInstr = VLXE_V_MASK in
      def PseudoVLXE_V_MASK_M#vlmul
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, evr:$rs2, vectormask:$mask,
                    GPR:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;
  }

  let mayLoad = 0, mayStore = 1, hasSideEffects = 0,
      usesCustomInserter = 1,
      VLMul = vlmul in
  {
    let Uses = [VL, VTYPE],
        VLIndex = 2, SEWIndex = 3, MergeOpIndex = -1,
        BaseInstr = VSE_V in
      def PseudoVSE_V_M#vlmul
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, GPR:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;

    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE],
        VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
        BaseInstr = VSE_V_MASK in
      def PseudoVSE_V_MASK_M#vlmul
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, vectormask:$mask, GPR:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Uses = [VL, VTYPE],
        VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
        BaseInstr = VSSE_V in
      def PseudoVSSE_V_M#vlmul
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, GPR:$rs2, GPR:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;

    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE],
        VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
        BaseInstr = VSSE_V_MASK in
      def PseudoVSSE_V_MASK_M#vlmul
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, GPR:$rs2, vectormask:$mask, GPR:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Uses = [VL, VTYPE],
        VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
        BaseInstr = VSXE_V in
      def PseudoVSXE_V_M#vlmul
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, evr:$rs2, GPR:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;

    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE],
        VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
        BaseInstr = VSXE_V_MASK in
      def PseudoVSXE_V_MASK_M#vlmul
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, evr:$rs2, vectormask:$mask, GPR:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo;
  }
}

}

//===----------------------------------------------------------------------===//
// Patterns. Essential
//===----------------------------------------------------------------------===//

multiclass pat_load_store<LLVMType type,
                          string load_instr_name,
                          string store_instr_name,
                          int sew,
                          int vlmul,
                          EPIVReg reg_class>
{
  // Load
  def : Pat<(type (load GPR:$rs1)),
            (!cast<Instruction>(load_instr_name#"_M"#vlmul)
             GPR:$rs1, /* vl */ X0, sew)>;
  def : Pat<(type (load AddrFI:$rs1)),
            (!cast<Instruction>(load_instr_name#"_M"#vlmul)
             AddrFI:$rs1, /* vl */ X0, sew)>;

  // Store
  def : Pat<(store type:$rs2, GPR:$rs1),
            (!cast<Instruction>(store_instr_name#"_M"#vlmul)
             reg_class:$rs2, GPR:$rs1, /* vl */ X0, sew)>;
  def : Pat<(store type:$rs2, AddrFI:$rs1),
            (!cast<Instruction>(store_instr_name#"_M"#vlmul)
             reg_class:$rs2, AddrFI:$rs1, /* vl */ X0, sew)>;
}

// Extra patterns for truncating loads and extending stores of mask types
multiclass pat_load_store_mask<LLVMType type,
                               LLVMType integer_type,
                               string load_instr_name,
                               string store_instr_name,
                               int sew,
                               int vlmul,
                               EPIVReg reg_class>
{
  // Load
  def : Pat<(type (trunc (integer_type (load GPR:$rs1)))),
            (!cast<Instruction>(load_instr_name#"_M"#vlmul)
             GPR:$rs1, /* vl */ X0, sew)>;
  def : Pat<(type (trunc (integer_type (load AddrFI:$rs1)))),
            (!cast<Instruction>(load_instr_name#"_M"#vlmul)
             AddrFI:$rs1, /* vl */ X0, sew)>;

  // Store
  def : Pat<(store (integer_type (zext type:$rs2)), GPR:$rs1),
            (!cast<Instruction>(store_instr_name#"_M"#vlmul)
             reg_class:$rs2, GPR:$rs1, /* vl */ X0, sew)>;
  def : Pat<(store (integer_type (zext type:$rs2)), AddrFI:$rs1),
            (!cast<Instruction>(store_instr_name#"_M"#vlmul)
             reg_class:$rs2, AddrFI:$rs1, /* vl */ X0, sew)>;
}

let Predicates = [HasExtEPI] in {
  foreach vti = AllVectors in
  {
    defm : pat_load_store<vti.Vector, "PseudoVLE_V", "PseudoVSE_V", vti.SEW,
                          vti.RegClass.VLMul, vti.RegClass>;
  }

  foreach mti = AllMasks in
  {
    // Mask values are stored as vectors of bytes
    defm : pat_load_store_mask<mti.Mask, mti.IntegerVector,
                               "PseudoVLE_V", "PseudoVSE_V",
                               /* sew */ 8, /* vlmul */ 1, EPIVR>;
  }

  foreach vti = NoGroupIntegerVectors in {
     def : Pat<(vti.Vector (zext (vti.Mask EPIVR:$rs1))),
               (PseudoVMAND_MM_M1 EPIVR:$rs1, EPIVR:$rs1, X0, vti.SEW)>;
     // We don't have to do anything here because mask vectors only use the
     // LSB of each element.
     def : Pat<(vti.Mask (trunc (vti.Vector EPIVR:$rs1))),
               (COPY_TO_REGCLASS EPIVR:$rs1, EPIVR)>;
  }

  foreach vti = GroupIntegerVectors in {
     def : Pat<(vti.Mask (trunc (vti.Vector vti.RegClass:$rs1))),
               (!cast<Instruction>("PseudoVMSNE_VV_M"#vti.RegClass.VLMul)
                 (!cast<Instruction>("PseudoVAND_VI_M"#vti.RegClass.VLMul)
                   (vti.Vector vti.RegClass:$rs1), 1, X0, vti.SEW),
                 (!cast<Instruction>("PseudoVAND_VI_M"#vti.RegClass.VLMul)
                   (vti.Vector (IMPLICIT_DEF)), 0, X0, vti.SEW),
                 X0, vti.SEW)>;
     def : Pat<(vti.Vector (zext (vti.Mask V0))),
               (!cast<Instruction>("PseudoVMERGE_VIM_M"#vti.RegClass.VLMul)
                 (!cast<Instruction>("PseudoVAND_VI_M"#vti.RegClass.VLMul)
                   (vti.Vector (IMPLICIT_DEF)), 0, X0, vti.SEW),
                 1, (vti.Mask V0), X0, vti.SEW)>;
  }

  // FIXME Deal with {B,H,W} load/store (load + sext)
}

//===----------------------------------------------------------------------===//
// Patterns. Vector register configuration
//===----------------------------------------------------------------------===//

// Using the X0 register or a GPR with value 0 as operands for VSETVL[I] have
// different semantincs: X0 entails VL := VLMAX, while using a GPR with value 0
// entails VL := 0. Uses of a i64 0 constant may be selected into X0, and thus
// the following node is used to represent a register or constant that may hold
// a 0 value and shouldn't be selected into X0.
//
// Note: 'undef' is used as Opcode parameter, given that there is no 'wildcard'
// SDNode.
def NoX0 : SDNodeXForm<undef,
[{
    SDLoc DL(N);

    if (auto *C = dyn_cast<ConstantSDNode>(N)) {
      if (C->isNullValue()) {
        return SDValue(CurDAG->getMachineNode(RISCV::ADDI, DL, MVT::i64,
                           CurDAG->getRegister(RISCV::X0, MVT::i64),
                           CurDAG->getTargetConstant(0, DL, MVT::i64)), 0);
      }
    }

    return SDValue(N, 0);
}]>;

let Predicates = [HasExtEPI] in {

def : Pat<(int_experimental_vector_vscale), (PseudoVSCALE)>;

// FIXME: uimm5 here is not very precise. It should be uimm3 and uimm2 but we
// will probably want to wait for intrinsics with constant values constraints.
def : Pat<(int_epi_vsetvl GPR:$vl, uimm5:$sew, uimm5:$vmul),
          (VSETVLI (NoX0 GPR:$vl), uimm5:$sew, uimm5:$vmul)>;

// FIXME: uimm5 here is not very precise. It should be uimm3 and uimm2 but we
// will probably want to wait for intrinsics with constant values constraints.
def : Pat<(int_epi_vsetvlmax uimm5:$sew, uimm5:$vmul),
          (VSETVLI X0, uimm5:$sew, uimm5:$vmul)>;

}

//===----------------------------------------------------------------------===//
// Patterns. Arithmetic
//===----------------------------------------------------------------------===//

class swap_helper<dag Prefix,
                  dag A,
                  dag B,
                  dag Suffix,
                  bit swap> {
   dag Value = !con(
       Prefix,
       !if(swap, B, A),
       !if(swap, A, B),
       Suffix);
}

multiclass pat_intrinsic_binary_nomask<string intrinsic_name,
                                       string instruction_name,
                                       string kind,
                                       ValueType result_type,
                                       ValueType op1_type,
                                       ValueType op2_type,
                                       int sew,
                                       int vlmul,
                                       EPIVReg op1_reg_class,
                                       RegisterClass op2_reg_class, // FIXME dagoperand?
                                       bit swap = 0>
{
  foreach instruction = [!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)] in
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2),
                          (i64 GPR:$vl))),
            swap_helper<
              (instruction), // Empty prefix
              (instruction
               op1_reg_class:$rs1),
              // Floating point instructions with a scalar operand expect such
              // operand to be in a register of class FPR64. When dealing with
              // the f32 variant of such instructions we need to promote the
              // FPR32 subregister to the FPR64 base register
              (instruction
               !if(!eq(!cast<string>(op2_reg_class),
                       !cast<string>(FPR32)),
                   (SUBREG_TO_REG (i32 -1), FPR32:$rs2, sub_32),
                   (op2_type op2_reg_class:$rs2))),
              (instruction
               (NoX0 GPR:$vl),
               sew),
              swap>.Value>;
}

multiclass pat_intrinsic_binary<string intrinsic_name,
                                string instruction_name,
                                string kind,
                                ValueType result_type,
                                ValueType op1_type,
                                ValueType op2_type,
                                ValueType mask_type,
                                int sew,
                                int vlmul,
                                EPIVReg result_reg_class,
                                EPIVReg op1_reg_class,
                                RegisterClass op2_reg_class, // FIXME dagoperand?
                                bit swap = 0>
{
  defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, kind,
                                     result_type, op1_type, op2_type, sew,
                                     vlmul, op1_reg_class, op2_reg_class,
                                     swap>;

  foreach instruction = [!cast<Instruction>(instruction_name#_#kind#"_MASK_M"#vlmul)] in
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                          (result_type result_reg_class:$merge),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2),
                          (mask_type V0),
                          (i64 GPR:$vl))),
            swap_helper<
              (instruction result_reg_class:$merge),
              (instruction
               op1_reg_class:$rs1),
              // Floating point instructions with a scalar operand expect such
              // operand to be in a register of class FPR64. When dealing with
              // the f32 variant of such instructions we need to promote the
              // FPR32 subregister to the FPR64 base register
              (instruction
               !if(!eq(!cast<string>(op2_reg_class),
                       !cast<string>(FPR32)),
                   (SUBREG_TO_REG (i32 -1), FPR32:$rs2, sub_32),
                   (op2_type op2_reg_class:$rs2))),
              (instruction
               (mask_type V0),
               (NoX0 GPR:$vl),
               sew),
              swap>.Value>;
}

multiclass pat_intrinsic_binary_mask_in<string intrinsic_name,
                                        string instruction_name,
                                        string kind,
                                        ValueType result_type,
                                        ValueType op1_type,
                                        ValueType op2_type,
                                        ValueType mask_type,
                                        int sew,
                                        int vlmul,
                                        EPIVReg result_reg_class,
                                        EPIVReg op1_reg_class,
                                        RegisterClass op2_reg_class, // FIXME dagoperand?
                                        bit swap = 0>
{
  foreach instruction = [!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)] in
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2),
                          (mask_type V0),
                          (i64 GPR:$vl))),
            swap_helper<
              (instruction), // empty prefix
              (instruction
               op1_reg_class:$rs1),
              // Floating point instructions with a scalar operand expect such
              // operand to be in a register of class FPR64. When dealing with
              // the f32 variant of such instructions we need to promote the
              // FPR32 subregister to the FPR64 base register
              (instruction
               !if(!eq(!cast<string>(op2_reg_class),
                       !cast<string>(FPR32)),
                   (SUBREG_TO_REG (i32 -1), FPR32:$rs2, sub_32),
                   (op2_type op2_reg_class:$rs2))),
              (instruction
               (mask_type V0),
               (NoX0 GPR:$vl),
               sew),
              swap>.Value>;
}

multiclass pat_intrinsic_binary_int_v_vv<string intrinsic_name,
                                         string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, vti.RegClass>;
  }
}

multiclass pat_intrinsic_binary_int_v_vv_vx<string intrinsic_name,
                                            string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_v_vx_vi<string intrinsic_name,
                                            string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_v_vv_vx_vi<string intrinsic_name,
                                               string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_v_vvm_vxm_vim<string intrinsic_name,
                                                  string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VVM",
                                        vti.Vector, vti.Vector, vti.Vector,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass,
                                        vti.RegClass>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VXM",
                                        vti.Vector, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VIM",
                                        vti.Vector, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass, simm5>;
  }
}


multiclass pat_intrinsic_binary_int_v_vs<string intrinsic_name,
                                         string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VS",
                                vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, vti.RegClass>;
  }
}

multiclass pat_intrinsic_binary_int_w_vv_vx<string intrinsic_name,
                                            string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  foreach vti = [vtiToWti.Vti] in
  foreach wti = [vtiToWti.Wti] in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                wti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, wti.RegClass,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                wti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, wti.RegClass,
                                vti.RegClass, GPR>;
  }
}

multiclass pat_intrinsic_binary_relational_int_m_vv_vx_vi<string intrinsic_name,
                                                          string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, EPIVR,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, EPIVR, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, EPIVR, vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_relational_int_m_vv_vx_vu<string intrinsic_name,
                                                          string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, EPIVR,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, EPIVR, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VU",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, EPIVR, vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_relational_int_m_vv_vx<string intrinsic_name,
                                                       string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, EPIVR,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, EPIVR, vti.RegClass, GPR>;
  }
}

multiclass pat_intrinsic_binary_relational_int_m_vx_vi<string intrinsic_name,
                                                       string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, EPIVR, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, EPIVR, vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_relational_int_m_vx_vu<string intrinsic_name,
                                                       string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, EPIVR, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VU",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, EPIVR, vti.RegClass, uimm5>;
  }
}

multiclass pat_intrinsic_binary_relational_int_v_vv_swapped<string intrinsic_name,
                                                            string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, EPIVR,
                                vti.RegClass, vti.RegClass, /* swapped */ 1>;
  }
}

//multiclass pat_intrinsic_binary_with_mask_input_vv_nomask<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType result_type>
//{
//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), (op_type EPIVR:$rs2), (op_type V0))),
//            (!cast<Instruction>(instruction_name # "_VV")
//               EPIVR:$rs1, EPIVR:$rs2)>;
//}

//multiclass pat_intrinsic_binary_with_mask_input_vx_nomask<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType result_type>
//{
//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), GPR:$rs2, (op_type V0))),
//            (!cast<Instruction>(instruction_name # "_VX")
//               EPIVR:$rs1, GPR:$rs2)>;
//}

//multiclass pat_intrinsic_binary_with_mask_input_vi_nomask<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType result_type>
//{
//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), simm5:$imm5, (op_type V0))),
//            (!cast<Instruction>(instruction_name # "_VI")
//               EPIVR:$rs1, simm5:$imm5)>;
//}

//multiclass pat_intrinsic_binary_with_mask_input_int_vv_vx_vi_nomask<string intrinsic_name,
//                                    string instruction_name>
//{
//  foreach vtp = AllIntegerVectors
//  in {
//    defm : pat_intrinsic_binary_with_mask_input_vv_nomask<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
//    defm : pat_intrinsic_binary_with_mask_input_vx_nomask<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
//    defm : pat_intrinsic_binary_with_mask_input_vi_nomask<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
//  }
//}

//multiclass pat_intrinsic_binary_with_mask_input_int_vv_vx_nomask<string intrinsic_name,
//                                    string instruction_name>
//{
//  foreach vtp = AllIntegerVectors
//  in {
//    defm : pat_intrinsic_binary_with_mask_input_vv_nomask<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
//    defm : pat_intrinsic_binary_with_mask_input_vx_nomask<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
//  }
//}

multiclass pat_intrinsic_binary_fp_v_vv<string intrinsic_name,
                                        string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fvti.Vector, fvti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fvti.RegClass, fvti.RegClass, fvti.RegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vf<string intrinsic_name,
                                        string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Vector, fvti.Vector, fvti.Scalar,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fvti.RegClass, fvti.RegClass,
                                fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vs<string intrinsic_name,
                                        string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VS",
                                fvti.Vector, fvti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fvti.RegClass, fvti.RegClass, fvti.RegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vv_vf<string intrinsic_name,
                                           string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fvti.Vector, fvti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fvti.RegClass, fvti.RegClass, fvti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Vector, fvti.Vector, fvti.Scalar,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fvti.RegClass, fvti.RegClass,
                                fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vvm<string intrinsic_name,
                                         string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VVM",
                                        fvti.Vector, fvti.Vector, fvti.Vector,
                                        fvti.Mask, fvti.SEW,
                                        fvti.RegClass.VLMul, fvti.RegClass,
                                        fvti.RegClass, fvti.RegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vfm<string intrinsic_name,
                                         string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VFM",
                                        fvti.Vector, fvti.Vector, fvti.Scalar,
                                        fvti.Mask, fvti.SEW,
                                        fvti.RegClass.VLMul, fvti.RegClass,
                                        fvti.RegClass, fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_w_vv_vf<string intrinsic_name,
                                           string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  foreach fvti = [fvtiToFWti.FVti] in
  foreach fwti = [fvtiToFWti.FWti] in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fwti.Vector, fvti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fwti.RegClass, fvti.RegClass, fvti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fwti.Vector, fvti.Vector, fvti.Scalar,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fwti.RegClass, fvti.RegClass,
                                fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_relational_fp_m_vv_vf<string intrinsic_name,
                                                      string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fvti.Mask, fvti.Vector, fvti.Vector, fvti.Mask,
                                fvti.SEW, fvti.RegClass.VLMul, EPIVR,
                                fvti.RegClass, fvti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Mask, fvti.Vector, fvti.Scalar, fvti.Mask,
                                fvti.SEW, fvti.RegClass.VLMul, EPIVR,
                                fvti.RegClass, fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_relational_fp_m_vf<string intrinsic_name,
                                                   string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Mask, fvti.Vector, fvti.Scalar, fvti.Mask,
                                fvti.SEW, fvti.RegClass.VLMul, EPIVR,
                                fvti.RegClass, fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_relational_fp_m_vv_swapped<string intrinsic_name,
                                                           string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fvti.Mask, fvti.Vector, fvti.Vector, fvti.Mask,
                                fvti.SEW, fvti.RegClass.VLMul, EPIVR,
                                fvti.RegClass, fvti.RegClass, /* swapped */ 1>;
  }
}

multiclass pat_intrinsic_binary_m_mm<string intrinsic_name,
                                     string instruction_name>
{
  foreach mti = AllMasks in
  {
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "MM",
                                       mti.Mask, mti.Mask, mti.Mask, mti.SEW,
                                       mti.VLMul, EPIVR, EPIVR>;
  }
}

multiclass pat_intrinsic_ternary<string intrinsic_name,
                                 string instruction_name,
                                 string kind,
                                 ValueType result_type,
                                 ValueType op1_type,
                                 ValueType op2_type,
                                 ValueType mask_type,
                                 int sew,
                                 int vlmul,
                                 EPIVReg result_reg_class,
                                 EPIVReg op1_reg_class,
                                 EPIVReg op2_reg_class> // FIXME Registerclass, dagoperand?
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (result_type result_reg_class:$rs3),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2),
                          (i64 GPR:$vl))),
            (!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)
             result_reg_class:$rs3,
             // Floating point instructions with a scalar operand expect such
             // operand to be in a register of class FPR64. When dealing with
             // the f32 variant of such instructions we need to promote the
             // FPR32 subregister to the FPR64 base register
             !if(!eq(!cast<string>(op1_reg_class),
                     !cast<string>(FPR32)),
                 (SUBREG_TO_REG (i32 -1), FPR32:$rs1, sub_32),
                 (op1_type op1_reg_class:$rs1)),
             op2_reg_class:$rs2,
             (NoX0 GPR:$vl),
             sew)>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                          (result_type result_reg_class:$rs3),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2),
                          (mask_type V0),
                          (i64 GPR:$vl))),
            (!cast<Instruction>(instruction_name#_#kind#"_MASK_M"#vlmul)
             result_reg_class:$rs3,
             // Floating point instructions with a scalar operand expect such
             // operand to be in a register of class FPR64. When dealing with
             // the f32 variant of such instructions we need to promote the
             // FPR32 subregister to the FPR64 base register
             !if(!eq(!cast<string>(op1_reg_class),
                     !cast<string>(FPR32)),
                 (SUBREG_TO_REG (i32 -1), FPR32:$rs1, sub_32),
                 (op1_type op1_reg_class:$rs1)),
             op2_reg_class:$rs2,
             (mask_type V0),
             (NoX0 GPR:$vl), sew)>;
}

multiclass pat_intrinsic_ternary_int_v_vv_vx<string intrinsic_name,
                                             string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_ternary<intrinsic_name, instruction_name, "VV",
                                 vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                 vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                 vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_ternary<intrinsic_name, instruction_name, "VX",
                                 vti.Vector, XLenVT, vti.Vector, vti.Mask,
                                 vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                 GPR, vti.RegClass>;
  }
}

multiclass pat_intrinsic_ternary_fp_v_vv_vf<string intrinsic_name,
                                            string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_ternary<intrinsic_name, instruction_name, "VV",
                                 fvti.Vector, fvti.Vector, fvti.Vector,
                                 fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                 fvti.RegClass, fvti.RegClass, fvti.RegClass>;
    defm : pat_intrinsic_ternary<intrinsic_name, instruction_name, "VF",
                                 fvti.Vector, fvti.Scalar, fvti.Vector,
                                 fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                 fvti.RegClass, fvti.ScalarRegClass,
                                 fvti.RegClass>;
  }
}

//multiclass pat_intrinsic_binary_wv_nomask<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType result_type>
//{
//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), (op_type EPIVR:$rs2))),
//            (!cast<Instruction>(instruction_name # "_WV")
//               EPIVR:$rs1, EPIVR:$rs2)>;
//}

//multiclass pat_intrinsic_binary_wv<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType result_type>
//{
//  defm "" : pat_intrinsic_binary_wv_nomask<intrinsic_name, instruction_name,
//                                           op_type, result_type>;

//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
//                         (op_type EPIVR:$rs1),
//                         (op_type EPIVR:$rs2), V0)),
//            (!cast<Instruction>(instruction_name # "_WV_MASK")
//               EPIVR:$rs1, EPIVR:$rs2, vmask_only_true.Value)>;
//}

//multiclass pat_intrinsic_binary_wf_nomask<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType op_scalar,
//                                   ValueType result_type>
//{
//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), (op_scalar EPIFPR:$rs2))),
//            (!cast<Instruction>(instruction_name # "_WF")
//               EPIVR:$rs1, EPIFPR:$rs2)>;
//}

//multiclass pat_intrinsic_binary_wf<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType op_scalar,
//                                   ValueType result_type>
//{
//  defm "" : pat_intrinsic_binary_wf_nomask<intrinsic_name,
//                   instruction_name, op_type, op_scalar, result_type>;

//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
//                         (op_type EPIVR:$rs1),
//                         (op_scalar EPIFPR:$rs2), V0)),
//            (!cast<Instruction>(instruction_name # "_WF_MASK")
//               EPIVR:$rs1, EPIFPR:$rs2, vmask_only_true.Value)>;
//}

//multiclass pat_intrinsic_binary_fp_wv_wf<string intrinsic_name,
//                                         string instruction_name>
//{
//  foreach vtp = AllFloatVectors
//  in {
//    defm : pat_intrinsic_binary_wv<intrinsic_name, instruction_name,
//                                          vtp.Vector, vtp.Vector>;
//    defm : pat_intrinsic_binary_wf<intrinsic_name, instruction_name,
//                                          vtp.Vector, vtp.Scalar, vtp.Vector>;
//  }
//}


//multiclass pat_intrinsic_binary_wx_nomask<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType result_type>
//{
//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), GPR:$rs2)),
//            (!cast<Instruction>(instruction_name # "_WX")
//               EPIVR:$rs1, GPR:$rs2)>;
//}

//multiclass pat_intrinsic_binary_wx<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType result_type>
//{
//  defm "" : pat_intrinsic_binary_wx_nomask<intrinsic_name, instruction_name,
//                                           op_type, result_type>;

//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
//                         (op_type EPIVR:$rs1),
//                         GPR:$rs2, V0)),
//            (!cast<Instruction>(instruction_name # "_WX_MASK")
//               EPIVR:$rs1, GPR:$rs2, vmask_only_true.Value)>;
//}

//multiclass pat_intrinsic_binary_int_wv_wx<string intrinsic_name,
//                                    string instruction_name>
//{
//  foreach vtp = AllIntegerVectors
//  in {
//    defm : pat_intrinsic_binary_wv<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
//    defm : pat_intrinsic_binary_wx<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
//  }
//}

multiclass pat_intrinsic_binary_any_and_int_v_vx<string intrinsic_name,
                                                 string instruction_name>
{
  foreach vti = AllVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, GPR>;
  }
}

multiclass pat_intrinsic_binary_any_and_int_v_vv_vx_vi<string intrinsic_name,
                                                     string instruction_name>
{
  foreach vti = AllVectors in
  foreach ivti = [GetIntVectorTypeInfo<vti>.Vti] in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Vector, vti.Vector, ivti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_any_and_int_v_vx_vu<string intrinsic_name,
                                                    string instruction_name>
{
  foreach vti = AllVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VU",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, uimm5>;
  }
}

//class ValueTypePair<ValueType ty1, ValueType ty2>
//{
//   ValueType First = ty1;
//   ValueType Second = ty2;
//}

//class ValueTypeVarList<list<ValueTypePair> VList>
//{
//  list<ValueTypePair> Value = VList;
//}

//def SameSizePairs : ValueTypeVarList<
//  [ ValueTypePair<nxv1i32, nxv1f32>,
//    ValueTypePair<nxv1i64, nxv1f64> ]>;

//multiclass pat_conversions<string instruction,
//                           string intrinsic,
//                           list<ValueTypePair> pairs>
//{

//foreach vtp = pairs in
//{
//def : Pat<(vtp.First (!cast<Intrinsic>("int_epi_" # intrinsic) (vtp.Second EPIVR:$rs1))),
//          (!cast<Instruction>(instruction # "_V") EPIVR:$rs1)>;
//def : Pat<(vtp.First (!cast<Intrinsic>("int_epi_" # intrinsic # "_mask") (vtp.Second EPIVR:$rs1), V0)),
//          (!cast<Instruction>(instruction # "_V_MASK") EPIVR:$rs1, vmask_only_true.Value)>;

//}

//}

multiclass pat_intrinsic_unary<string intrinsic_name,
                               string instruction_name,
                               string kind,
                               ValueType result_type,
                               ValueType op1_type,
                               ValueType mask_type,
                               int sew,
                               int vlmul,
                               EPIVReg result_reg_class,
                               EPIVReg op1_reg_class> // FIXME Registerclass, dagoperand?
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (i64 GPR:$vl))),
            (!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)
             op1_reg_class:$rs1,
             (NoX0 GPR:$vl), sew)>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                          (result_type result_reg_class:$merge),
                          (op1_type op1_reg_class:$rs1),
                          (mask_type V0),
                          (i64 GPR:$vl))),
            (!cast<Instruction>(instruction_name#_#kind#"_MASK_M"#vlmul)
             result_reg_class:$merge,
             op1_reg_class:$rs1,
             (mask_type V0),
             (NoX0 GPR:$vl), sew)>;
}

multiclass pat_intrinsic_unary_int_w_v<string intrinsic_name,
                                       string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  foreach vti = [vtiToWti.Vti] in
  foreach wti = [vtiToWti.Wti] in
  {
    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               wti.Vector, vti.Vector, wti.Mask, vti.SEW,
                               vti.RegClass.VLMul, wti.RegClass, vti.RegClass>;
  }
}

multiclass pat_intrinsic_unary_int_v_w<string intrinsic_name,
                                       string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  foreach vti = [vtiToWti.Vti] in
  foreach wti = [vtiToWti.Wti] in
  {
    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               vti.Vector, wti.Vector, vti.Mask, vti.SEW,
                               vti.RegClass.VLMul, vti.RegClass, wti.RegClass>;
  }
}


multiclass pat_intrinsic_unary_fp_w_v<string intrinsic_name,
                                      string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  foreach fvti = [fvtiToFWti.FVti] in
  foreach fwti = [fvtiToFWti.FWti] in
  {
    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               fwti.Vector, fvti.Vector, fwti.Mask, fvti.SEW,
                               fvti.RegClass.VLMul, fwti.RegClass,
                               fvti.RegClass>;
  }
}

multiclass pat_intrinsic_unary_int_w_fp_v<string intrinsic_name,
                                          string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  foreach fvti = [fvtiToFWti.FVti] in
  foreach iwti = [GetIntVectorTypeInfo<fvtiToFWti.FWti>.Vti] in
  {
    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               iwti.Vector, fvti.Vector, iwti.Mask, fvti.SEW,
                               fvti.RegClass.VLMul, iwti.RegClass,
                               fvti.RegClass>;
  }
}

multiclass pat_intrinsic_unary_fp_w_int_v<string intrinsic_name,
                                          string instruction_name>
{
  foreach vtiToWti = AllWideableIntToFloatVectors in
  foreach vti = [vtiToWti.Vti] in
  foreach fwti = [GetFloatVectorTypeInfo<vtiToWti.Wti>.FVti] in
  {
    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               fwti.Vector, vti.Vector, fwti.Mask, vti.SEW,
                               vti.RegClass.VLMul, fwti.RegClass, vti.RegClass>;
  }
}

multiclass pat_intrinsic_unary_fp_v_w<string intrinsic_name,
                                      string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  foreach fvti = [fvtiToFWti.FVti] in
  foreach fwti = [fvtiToFWti.FWti] in
  {
    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               fvti.Vector, fwti.Vector, fvti.Mask, fvti.SEW,
                               fvti.RegClass.VLMul, fvti.RegClass,
                               fwti.RegClass>;
  }
}

multiclass pat_intrinsic_unary_int_v_fp_w<string intrinsic_name,
                                          string instruction_name>
{
  foreach vtiToWti = AllWideableIntToFloatVectors in
  foreach vti = [vtiToWti.Vti] in
  foreach fwti = [GetFloatVectorTypeInfo<vtiToWti.Wti>.FVti] in
  {
    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               vti.Vector, fwti.Vector, vti.Mask, vti.SEW,
                               vti.RegClass.VLMul, vti.RegClass, fwti.RegClass>;
  }
}

multiclass pat_intrinsic_unary_fp_v_int_w<string intrinsic_name,
                                          string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  foreach fvti = [fvtiToFWti.FVti] in
  foreach iwti = [GetIntVectorTypeInfo<fvtiToFWti.FWti>.Vti] in
  {
    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               fvti.Vector, iwti.Vector, fvti.Mask, fvti.SEW,
                               fvti.RegClass.VLMul, fvti.RegClass,
                               iwti.RegClass>;
  }
}

//// We are missing cases but are of sizes that won't work
//def WidenedSize : ValueTypeVarList<
//  [ ValueTypePair<nxv1i64, nxv1f32>,
//    ValueTypePair<nxv1f32, nxv1i16>,
//    ValueTypePair<nxv1f64, nxv1i32> ]>;

//// Narrowing
//// TODO: This list is the inverse of the one above
//def NarrowedSize : ValueTypeVarList<
//  [ ValueTypePair<nxv1f32, nxv1i64>,
//    ValueTypePair<nxv1i16, nxv1f32>,
//    ValueTypePair<nxv1i32, nxv1f64> ]>;

//def FloatWidenedSize : ValueTypeVarList<
//  [ ValueTypePair<nxv1f64, nxv1f32> ]>;

//def FloatNarrowedSize : ValueTypeVarList<
//  [ ValueTypePair<nxv1f32, nxv1f64> ]>;

let Predicates = [HasExtEPI] in {

defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vadd", "PseudoVADD">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vsub", "PseudoVSUB">;
defm "" : pat_intrinsic_binary_int_v_vx_vi<"int_epi_vrsub", "PseudoVRSUB">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vminu", "PseudoVMINU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmin", "PseudoVMIN">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmaxu", "PseudoVMAXU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmax", "PseudoVMAX">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vand", "PseudoVAND">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vor", "PseudoVOR">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vxor", "PseudoVXOR">;

defm "" : pat_intrinsic_binary_any_and_int_v_vv_vx_vi<"int_epi_vrgather", "PseudoVRGATHER">;
defm "" : pat_intrinsic_binary_any_and_int_v_vx_vu<"int_epi_vslideup", "PseudoVSLIDEUP">;
defm "" : pat_intrinsic_binary_any_and_int_v_vx_vu<"int_epi_vslidedown", "PseudoVSLIDEDOWN">;

//defm "" : pat_intrinsic_binary_with_mask_input_int_vv_vx_vi_nomask<"int_epi_vadc", "VADC">;
//defm "" : pat_intrinsic_binary_with_mask_input_int_vv_vx_nomask<"int_epi_vsbc", "VSBC">;

defm "" : pat_intrinsic_binary_relational_int_m_vv_vx_vi<"int_epi_vmseq", "PseudoVMSEQ">;
defm "" : pat_intrinsic_binary_relational_int_m_vv_vx_vi<"int_epi_vmsne", "PseudoVMSNE">;
defm "" : pat_intrinsic_binary_relational_int_m_vv_vx<"int_epi_vmsltu", "PseudoVMSLTU">;
defm "" : pat_intrinsic_binary_relational_int_m_vv_vx<"int_epi_vmslt", "PseudoVMSLT">;
defm "" : pat_intrinsic_binary_relational_int_m_vv_vx_vu<"int_epi_vmsleu", "PseudoVMSLEU">;
defm "" : pat_intrinsic_binary_relational_int_m_vv_vx_vi<"int_epi_vmsle", "PseudoVMSLE">;

defm "" : pat_intrinsic_binary_relational_int_m_vx_vu<"int_epi_vmsgtu", "PseudoVMSGTU">;
// Select (int_epi_vmsgtu reg:$rs1, reg:$rs2) as (PseudoVMSLTU reg:$rs2, reg:$rs1)
defm "" : pat_intrinsic_binary_relational_int_v_vv_swapped<"int_epi_vmsgtu", "PseudoVMSLTU">;

defm "" : pat_intrinsic_binary_relational_int_m_vx_vi<"int_epi_vmsgt", "PseudoVMSGT">;
// Select (int_epi_vmsgt reg:$rs1, reg:$rs2) as (PseudoVMSLT reg:$rs2, reg:$rs1)
defm "" : pat_intrinsic_binary_relational_int_v_vv_swapped<"int_epi_vmsgt", "PseudoVMSLT">;

defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsaddu", "PseudoVSADDU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsadd", "PseudoVSADD">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vssubu", "PseudoVSSUBU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vssub", "PseudoVSSUB">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vaadd", "PseudoVAADD">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsll", "PseudoVSLL">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vasub", "PseudoVASUB">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vsmul", "PseudoVSMUL">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsrl", "PseudoVSRL">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsra", "PseudoVSRA">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vssrl", "PseudoVSSRL">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vssra", "PseudoVSSRA">;

defm "" : pat_intrinsic_binary_int_v_vvm_vxm_vim<"int_epi_vmerge", "PseudoVMERGE">;

//// FIXME: These patterns are wrong
//defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vnsrl", "VNSRL">;
//defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vnsra", "VNSRA">;
//defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vnclipu", "VNCLIPU">;
//defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vnclip", "VNCLIP">;

//// FIXME: These patterns are wrong
//defm "" : pat_intrinsic_binary_int_vs<"int_epi_vwredsumu", "VWREDSUMU">;
//defm "" : pat_intrinsic_binary_int_vs<"int_epi_vwredsum", "VWREDSUM">;

defm "" : pat_intrinsic_binary_int_v_vv<"int_epi_vdotu", "PseudoVDOTU">;
defm "" : pat_intrinsic_binary_int_v_vv<"int_epi_vdot", "PseudoVDOT">;

//// FIXME: These patterns are wrong
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmaccu", "VWSMACCU">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmacc", "VWSMACC">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmsacu", "VWSMSACU">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmsac", "VWSMSAC">;

defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredsum", "PseudoVREDSUM">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredand", "PseudoVREDAND">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredor", "PseudoVREDOR">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredxor", "PseudoVREDXOR">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredminu", "PseudoVREDMINU">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredmin", "PseudoVREDMIN">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredmaxu", "PseudoVREDMAXU">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredmax", "PseudoVREDMAX">;

// We only handle nxv1i64 variant here, as nxv2i32, nxv4i16 and nxv8i8
// intrinsics have an illegal result type (i8, i16, i32) and are handled
// programatically
foreach vti = [Vtype1xi64] in
  def : Pat<(int_epi_vext_x_v (vti.Vector vti.RegClass:$rs2), GPR:$rs1),
            (PseudoVEXT_X_V $rs2, $rs1, vti.SEW)>;

foreach vti = NoGroupIntegerVectors in
  def : Pat<(vti.Vector (int_epi_vmv_s_x GPR:$rs2, GPR:$vl)),
            (PseudoVMV_S_X $rs2, (NoX0 GPR:$vl), vti.SEW)>;

defm "" : pat_intrinsic_binary_any_and_int_v_vx<"int_epi_vslide1up", "PseudoVSLIDE1UP">;
defm "" : pat_intrinsic_binary_any_and_int_v_vx<"int_epi_vslide1down", "PseudoVSLIDE1DOWN">;

foreach mti = AllMasks in {
  def : Pat<(int_epi_vmpopc (mti.Mask EPIVR:$rs2), GPR:$vl),
            (!cast<Instruction>("PseudoVMPOPC_M_M"#mti.VLMul) $rs2,
              (NoX0 GPR:$vl), mti.SEW)>;
  def : Pat<(int_epi_vmpopc_mask (mti.Mask EPIVR:$rs2), (mti.Mask V0), GPR:$vl),
            (!cast<Instruction>("PseudoVMPOPC_M_MASK_M"#mti.VLMul) $rs2,
              (mti.Mask V0), (NoX0 GPR:$vl), mti.SEW)>;

  def : Pat<(int_epi_vmfirst (mti.Mask EPIVR:$rs2), GPR:$vl),
            (!cast<Instruction>("PseudoVMFIRST_M_M"#mti.VLMul) $rs2,
              (NoX0 GPR:$vl), mti.SEW)>;
  def : Pat<(int_epi_vmfirst_mask (mti.Mask EPIVR:$rs2), (mti.Mask V0), GPR:$vl),
            (!cast<Instruction>("PseudoVMFIRST_M_MASK_M"#mti.VLMul) $rs2,
              (mti.Mask V0), (NoX0 GPR:$vl), mti.SEW)>;
}

//def : Pat<(int_epi_vmsbf EPIVR:$rs1),
//        (VMSBF_M $rs1)>;
//def : Pat<(int_epi_vmsbf_mask EPIVR:$rs1, V0),
//        (VMSBF_M_MASK $rs1, vmask_only_true.Value)>;

//def : Pat<(int_epi_vmsif EPIVR:$rs1),
//        (VMSIF_M $rs1)>;
//def : Pat<(int_epi_vmsif_mask EPIVR:$rs1, V0),
//        (VMSIF_M_MASK $rs1, vmask_only_true.Value)>;

//def : Pat<(int_epi_vmsof EPIVR:$rs1),
//        (VMSOF_M $rs1)>;
//def : Pat<(int_epi_vmsof_mask EPIVR:$rs1, V0),
//        (VMSOF_M_MASK $rs1, vmask_only_true.Value)>;

//foreach vtp = AllIntegerVectors in
//{
//def : Pat<(vtp.Vector (int_epi_viota EPIVR:$rs1)),
//        (VIOTA_M $rs1)>;
//def : Pat<(vtp.Vector (int_epi_viota_mask EPIVR:$rs1, V0)),
//        (VIOTA_M_MASK $rs1, vmask_only_true.Value)>;
//}

//foreach vtp = AllVectors in
//def : Pat<(vtp.Vector (int_epi_vcompress (vtp.Vector EPIVR:$rs2), (nxv1i1 EPIVR:$rs1))),
//          (VCOMPRESS_VM $rs2, $rs1)>;

defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmandnot", "PseudoVMANDNOT">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmand", "PseudoVMAND">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmor", "PseudoVMOR">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmxor", "PseudoVMXOR">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmornot", "PseudoVMORNOT">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmnand", "PseudoVMNAND">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmnor", "PseudoVMNOR">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmxnor", "PseudoVMXNOR">;

defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vdivu", "PseudoVDIVU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vdiv", "PseudoVDIV">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vremu", "PseudoVREMU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vrem", "PseudoVREM">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmulhu", "PseudoVMULHU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmul", "PseudoVMUL">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmulhsu", "PseudoVMULHSU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmulh", "PseudoVMULH">;

defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vmadd", "PseudoVMADD">;
defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vmsub", "PseudoVMSUB">;
defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vmacc", "PseudoVMACC">;
defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vmsac", "PseudoVMSAC">;

defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwaddu", "PseudoVWADDU">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwadd", "PseudoVWADD">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwsubu", "PseudoVWSUBU">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwsub", "PseudoVWSUB">;

//// FIXME: These patterns are wrong
//defm "" : pat_intrinsic_binary_int_wv_wx<"int_epi_vwaddu_w", "VWADDU">;
//defm "" : pat_intrinsic_binary_int_wv_wx<"int_epi_vwadd_w", "VWADD">;
//defm "" : pat_intrinsic_binary_int_wv_wx<"int_epi_vwsubu_w", "VWSUBU">;
//defm "" : pat_intrinsic_binary_int_wv_wx<"int_epi_vwsub_w", "VWSUB">;

defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwmulu", "PseudoVWMULU">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwmulsu", "PseudoVWMULSU">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwmul", "PseudoVWMUL">;

//// FIXME: These patterns are wrong
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmaccu", "VWMACCU">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmacc", "VWMACC">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmsacu", "VWMSACU">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmsac", "VWMSAC">;

defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfadd", "PseudoVFADD">;
defm "" : pat_intrinsic_binary_fp_v_vs<"int_epi_vfredsum", "PseudoVFREDSUM">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfsub", "PseudoVFSUB">;
defm "" : pat_intrinsic_binary_fp_v_vs<"int_epi_vfredosum", "PseudoVFREDOSUM">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfmin", "PseudoVFMIN">;
defm "" : pat_intrinsic_binary_fp_v_vs<"int_epi_vfredmin", "PseudoVFREDMIN">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfmax", "PseudoVFMAX">;
defm "" : pat_intrinsic_binary_fp_v_vs<"int_epi_vfredmax", "PseudoVFREDMAX">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfsgnj", "PseudoVFSGNJ">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfsgnjn", "PseudoVFSGNJN">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfsgnjx", "PseudoVFSGNJX">;

foreach fvti = NoGroupFloatVectors in
{
  def : Pat<(fvti.Scalar (int_epi_vfmv_f_s (fvti.Vector fvti.RegClass:$rs2),
                          GPR:$vl)),
             // Floating point instructions with a scalar result will always
             // generate the result in a register of class FPR64. When dealing
             // with the f32 variant of a pattern we need to extract the FPR32
             // subregister from the FPR64 register generated by the instruction
             !if(!eq(!cast<string>(fvti.ScalarRegClass),
                     !cast<string>(FPR32)),
                 (EXTRACT_SUBREG (PseudoVFMV_F_S
                                  $rs2, (NoX0 GPR:$vl), fvti.SEW),
                  sub_32),
                 (PseudoVFMV_F_S $rs2, (NoX0 GPR:$vl), fvti.SEW))>;

  def : Pat<(fvti.Vector (int_epi_vfmv_s_f
                          (fvti.Scalar fvti.ScalarRegClass:$rs2), GPR:$vl)),
            (PseudoVFMV_S_F
             // Floating point instructions with a scalar operand expect such
             // operand to be in a register of class FPR64. When dealing with
             // the f32 variant of such instructions we need to promote the
             // FPR32 subregister to the FPR64 base register
             !if(!eq(!cast<string>(fvti.ScalarRegClass),
                     !cast<string>(FPR32)),
                 (SUBREG_TO_REG (i32 -1), FPR32:$rs2, sub_32),
                 (fvti.Scalar fvti.ScalarRegClass:$rs2)),
             (NoX0 GPR:$vl), fvti.SEW)>;
}

defm "" : pat_intrinsic_binary_relational_fp_m_vv_vf<"int_epi_vmfeq", "PseudoVMFEQ">;
defm "" : pat_intrinsic_binary_relational_fp_m_vv_vf<"int_epi_vmfle", "PseudoVMFLE">;
defm "" : pat_intrinsic_binary_relational_fp_m_vv_vf<"int_epi_vmford", "PseudoVMFORD">;
defm "" : pat_intrinsic_binary_relational_fp_m_vv_vf<"int_epi_vmflt", "PseudoVMFLT">;
defm "" : pat_intrinsic_binary_relational_fp_m_vv_vf<"int_epi_vmfne", "PseudoVMFNE">;

defm "" : pat_intrinsic_binary_relational_fp_m_vf<"int_epi_vmfgt", "PseudoVMFGT">;
// Select (int_epi_vmfgt reg:$rs1, reg:$rs2) as (PseudoVMFLT reg:$rs2, reg:$rs1)
defm "" : pat_intrinsic_binary_relational_fp_m_vv_swapped<"int_epi_vmfgt", "PseudoVMFLT">;

defm "" : pat_intrinsic_binary_relational_fp_m_vf<"int_epi_vmfge", "PseudoVMFGE">;
// Select (int_epi_vmfge reg:$rs1, reg:$rs2) as (PseudoVMFLE reg:$rs2, reg:$rs1)
defm "" : pat_intrinsic_binary_relational_fp_m_vv_swapped<"int_epi_vmfge", "PseudoVMFLE">;

defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfdiv", "PseudoVFDIV">;
defm "" : pat_intrinsic_binary_fp_v_vf<"int_epi_vfrdiv", "PseudoVFRDIV">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfmul", "PseudoVFMUL">;

// There is no "vfmerge.vvm" so we select vmerge.vvm
defm "" : pat_intrinsic_binary_fp_v_vvm<"int_epi_vfmerge", "PseudoVMERGE">;
defm "" : pat_intrinsic_binary_fp_v_vfm<"int_epi_vfmerge", "PseudoVFMERGE">;

defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfmadd", "PseudoVFMADD">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfnmadd", "PseudoVFNMADD">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfmsub", "PseudoVFMSUB">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfnmsub", "PseudoVFNMSUB">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfmacc", "PseudoVFMACC">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfnmacc", "PseudoVFNMACC">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfmsac", "PseudoVFMSAC">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfnmsac", "PseudoVFNMSAC">;

defm "" : pat_intrinsic_binary_fp_w_vv_vf<"int_epi_vfwadd", "PseudoVFWADD">;
//defm "" : pat_intrinsic_binary_fp_vs<"int_epi_vfwredsum", "VFWREDSUM">;
defm "" : pat_intrinsic_binary_fp_w_vv_vf<"int_epi_vfwsub", "PseudoVFWSUB">;
//defm "" : pat_intrinsic_binary_fp_vs<"int_epi_vfwredosum", "VFWREDOSUM">;
//defm "" : pat_intrinsic_binary_fp_wv_wf<"int_epi_vfwadd_w", "VFWADD">;
//defm "" : pat_intrinsic_binary_fp_wv_wf<"int_epi_vfwsub_w", "VFWSUB">;
defm "" : pat_intrinsic_binary_fp_w_vv_vf<"int_epi_vfwmul", "PseudoVFWMUL">;
defm "" : pat_intrinsic_binary_fp_v_vv<"int_epi_vfdot", "PseudoVFDOT">;

//defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwmacc", "VFWMACC">;
//defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwnmacc", "VFWNMACC">;
//defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwmsac", "VFWMSAC">;
//defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwnmsac", "VFWNMSAC">;

foreach vti = AllFloatVectors in
  def : Pat<(vti.Vector (int_epi_vfsqrt (vti.Vector vti.RegClass:$rs2),
                         GPR:$vl)),
            (!cast<Instruction>("PseudoVFSQRT_V_M"#vti.RegClass.VLMul) $rs2,
             (NoX0 GPR:$vl), vti.SEW)>;

foreach vti = AllFloatVectors in
  def : Pat<(vti.Vector (int_epi_vfsqrt_mask (vti.Vector vti.RegClass:$merge),
                         (vti.Vector vti.RegClass:$rs2), (vti.Mask V0),
                         GPR:$vl)),
            (!cast<Instruction>("PseudoVFSQRT_V_MASK_M"#vti.RegClass.VLMul)
             $merge, $rs2, (vti.Mask V0), (NoX0 GPR:$vl), vti.SEW)>;

//foreach itp = [ nxv1i32, nxv1i64 ] in
//foreach vtp = AllFloatVectors in
//def : Pat<(itp (int_epi_vfclass (vtp.Vector EPIVR:$rs2))),
//          (VFCLASS_V $rs2)>;

//foreach itp = [ nxv1i32, nxv1i64 ] in
//foreach vtp = AllFloatVectors in
//def : Pat<(itp (int_epi_vfclass_mask (vtp.Vector EPIVR:$rs2), V0)),
//          (VFCLASS_V_MASK $rs2, vmask_only_true.Value)>;

multiclass pat_conversions_same_sew<string intrinsic_float_to_int,
                                    string instruction_float_to_int,
                                    string intrinsic_int_to_float,
                                    string instruction_int_to_float>
{
  foreach fvti = AllFloatVectors in
  foreach ivti = [GetIntVectorTypeInfo<fvti>.Vti] in
  foreach vlmul = [fvti.RegClass.VLMul] in
  {
    def : Pat<(ivti.Vector (!cast<Intrinsic>(intrinsic_float_to_int)
                            (fvti.Vector fvti.RegClass:$rs2),
                            GPR:$vl)),
              (!cast<Instruction>(instruction_float_to_int#"_M"#vlmul)
               $rs2, (NoX0 GPR:$vl), fvti.SEW)>;

    def : Pat<(ivti.Vector (!cast<Intrinsic>(intrinsic_float_to_int#"_mask")
                            (ivti.Vector fvti.RegClass:$merge),
                            (fvti.Vector fvti.RegClass:$rs2), (fvti.Mask V0),
                            GPR:$vl)),
              (!cast<Instruction>(instruction_float_to_int#"_MASK_M"#vlmul)
               $merge, $rs2, (fvti.Mask V0), (NoX0 GPR:$vl), fvti.SEW)>;

    def : Pat<(fvti.Vector (!cast<Intrinsic>(intrinsic_int_to_float)
                            (ivti.Vector fvti.RegClass:$rs2),
                            GPR:$vl)),
              (!cast<Instruction>(instruction_int_to_float#"_M"#vlmul)
               $rs2, (NoX0 GPR:$vl), fvti.SEW)>;

    def : Pat<(fvti.Vector (!cast<Intrinsic>(intrinsic_int_to_float#"_mask")
                            (fvti.Vector fvti.RegClass:$merge),
                            (ivti.Vector fvti.RegClass:$rs2), (fvti.Mask V0),
                            GPR:$vl)),
              (!cast<Instruction>(instruction_int_to_float#"_MASK_M"#vlmul)
               $merge, $rs2, (fvti.Mask V0), (NoX0 GPR:$vl), fvti.SEW)>;
  }
}


defm "" : pat_conversions_same_sew<"int_epi_vfcvt_xu_f", "PseudoVFCVT_XU_F_V",
                                   "int_epi_vfcvt_f_xu", "PseudoVFCVT_F_XU_V">;
defm "" : pat_conversions_same_sew<"int_epi_vfcvt_x_f", "PseudoVFCVT_X_F_V",
                                   "int_epi_vfcvt_f_x", "PseudoVFCVT_F_X_V">;

defm : pat_intrinsic_unary_int_w_fp_v<"int_epi_vfwcvt_xu_f", "PseudoVFWCVT_XU_F">;
defm : pat_intrinsic_unary_int_w_fp_v<"int_epi_vfwcvt_x_f", "PseudoVFWCVT_X_F">;

defm : pat_intrinsic_unary_fp_w_int_v<"int_epi_vfwcvt_f_xu", "PseudoVFWCVT_F_XU">;
defm : pat_intrinsic_unary_fp_w_int_v<"int_epi_vfwcvt_f_x", "PseudoVFWCVT_F_X">;

defm : pat_intrinsic_unary_fp_w_v<"int_epi_vfwcvt_f_f", "PseudoVFWCVT_F_F">;

defm : pat_intrinsic_unary_int_v_fp_w<"int_epi_vfncvt_xu_f", "PseudoVFNCVT_XU_F">;
defm : pat_intrinsic_unary_int_v_fp_w<"int_epi_vfncvt_x_f", "PseudoVFNCVT_X_F">;

defm : pat_intrinsic_unary_fp_v_int_w<"int_epi_vfncvt_f_xu", "PseudoVFNCVT_F_XU">;
defm : pat_intrinsic_unary_fp_v_int_w<"int_epi_vfncvt_f_x", "PseudoVFNCVT_F_X">;

defm : pat_intrinsic_unary_fp_v_w<"int_epi_vfncvt_f_f", "PseudoVFNCVT_F_F">;

multiclass pat_intrinsic_binary_int_w_vX0<string intrinsic_name,
                                          string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  foreach vti = [vtiToWti.Vti] in
  foreach wti = [vtiToWti.Wti] in
  foreach vlmul = [vti.RegClass.VLMul] in
  {
    def : Pat<(wti.Vector (!cast<Intrinsic>(intrinsic_name)
                           (vti.Vector vti.RegClass:$rs1),
                           (i64 GPR:$vl))),
              (!cast<Instruction>(instruction_name#"_M"#vlmul) // FIXME change when fix nameing in binary intrinsics
               vti.RegClass:$rs1,
               /* rs2 */ X0,
               (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(wti.Vector (!cast<Intrinsic>(intrinsic_name#"_mask")
                           (wti.Vector wti.RegClass:$merge),
                           (vti.Vector vti.RegClass:$rs1),
                           (wti.Mask V0),
                           (i64 GPR:$vl))),
              (!cast<Instruction>(instruction_name#"_MASK_M"#vlmul) // FIXME change when fix nameing in binary intrinsics
               wti.RegClass:$merge,
               vti.RegClass:$rs1,
               /* rs2 */ X0,
               (wti.Mask V0),
               (NoX0 GPR:$vl), vti.SEW)>;
  }
}

multiclass pat_intrinsic_binary_int_v_wX0<string intrinsic_name,
                                          string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  foreach vti = [vtiToWti.Vti] in
  foreach wti = [vtiToWti.Wti] in
  foreach vlmul = [vti.RegClass.VLMul] in
  {
    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic_name)
                           (wti.Vector wti.RegClass:$rs1),
                           (i64 GPR:$vl))),
              (!cast<Instruction>(instruction_name#"_M"#vlmul) // FIXME change when fix nameing in binary intrinsics
               wti.RegClass:$rs1,
               /* rs2 */ X0,
               (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic_name#"_mask")
                           (vti.Vector vti.RegClass:$merge),
                           (wti.Vector wti.RegClass:$rs1),
                           (vti.Mask V0),
                           (i64 GPR:$vl))),
              (!cast<Instruction>(instruction_name#"_MASK_M"#vlmul) // FIXME change when fix nameing in binary intrinsics
               vti.RegClass:$merge,
               wti.RegClass:$rs1,
               /* rs2 */ X0,
               (vti.Mask V0),
               (NoX0 GPR:$vl), vti.SEW)>;
  }
}

defm : pat_intrinsic_binary_int_w_vX0<"int_epi_vwcvt_xu_x", "PseudoVWADDU_VX">;
defm : pat_intrinsic_binary_int_w_vX0<"int_epi_vwcvt_x_x", "PseudoVWADD_VX">;

defm : pat_intrinsic_binary_int_v_wX0<"int_epi_vncvt_x_x", "PseudoVNSRL_VX">;

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (int_epi_vid GPR:$vl)),
            (!cast<Instruction>("PseudoVID_V_M"#vti.RegClass.VLMul)
             (NoX0 GPR:$vl), vti.SEW)>;
  def : Pat<(vti.Vector (int_epi_vid_mask (vti.Vector vti.RegClass:$merge),
                         (vti.Mask V0), GPR:$vl)),
            (!cast<Instruction>("PseudoVID_V_MASK_M"#vti.RegClass.VLMul)
             $merge, (vti.Mask V0), (NoX0 GPR:$vl), vti.SEW)>;
}

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (int_epi_vbroadcast GPR:$rs2, GPR:$vl)),
            (!cast<Instruction>("PseudoVMV_V_X_M"#vti.RegClass.VLMul)
             $rs2, (NoX0 GPR:$vl), vti.SEW)>;
  def : Pat<(vti.Vector (int_epi_vbroadcast simm5:$imm5, GPR:$vl)),
            (!cast<Instruction>("PseudoVMV_V_I_M"#vti.RegClass.VLMul)
             simm5:$imm5, (NoX0 GPR:$vl), vti.SEW)>;
}
foreach fvti = AllFloatVectors in {
  def : Pat<(fvti.Vector (int_epi_vbroadcast
                         (fvti.Scalar fvti.ScalarRegClass:$rs2), GPR:$vl)),
            (!cast<Instruction>("PseudoVFMV_V_F_M"#fvti.RegClass.VLMul)
             // Floating point instructions with a scalar operand expect such
             // operand to be in a register of class FPR64. When dealing with
             // the f32 variant of such instructions we need to promote the
             // FPR32 subregister to the FPR64 base register
             !if(!eq(!cast<string>(fvti.ScalarRegClass),
                     !cast<string>(FPR32)),
                 (SUBREG_TO_REG (i32 -1), FPR32:$rs2, sub_32),
                 (fvti.Scalar fvti.ScalarRegClass:$rs2)),
             $vl, fvti.SEW)>;
}
}

//===----------------------------------------------------------------------===//
// Patterns. Full vector arithmetic operations
//===----------------------------------------------------------------------===//

// FIXME: Add patterns that can select masked versions
// FIXME: Add patterns that can select scalar versions
multiclass pat_vop_binary<SDNode vop,
                          string instruction_name,
                          ValueType result_type,
                          ValueType op_type,
                          ValueType mask_type,
                          int sew,
                          int vlmul,
                          EPIVReg result_reg_class,
                          EPIVReg op_reg_class,
                          bit swap = 0>
{
  foreach instruction = [!cast<Instruction>(instruction_name#"_VV_M"#vlmul)] in
  def : Pat<(result_type (vop
                          (op_type op_reg_class:$rs1),
                          (op_type op_reg_class:$rs2))),
            swap_helper<
              (instruction), // Empty prefix
              (instruction op_reg_class:$rs1),
              (instruction op_reg_class:$rs2),
              (instruction X0, sew),
              swap>.Value>;
}

multiclass pat_vop_binary_common<SDNode vop,
                                 string instruction_name,
                                 list<VectorTypeInfo> vtilist>
{
  foreach vti = vtilist in
  {
    defm : pat_vop_binary<vop, instruction_name,
                          vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                          vti.RegClass.VLMul, vti.RegClass, vti.RegClass>;
  }
}

// FIXME: Add patterns that can select masked versions
multiclass pat_vop_ternary<SDNode vop,
                           string instruction_name,
                           ValueType result_type,
                           ValueType op1_type,
                           ValueType op2_type,
                           ValueType mask_type,
                           int sew,
                           int vlmul,
                           EPIVReg result_reg_class,
                           EPIVReg op1_reg_class,
                           EPIVReg op2_reg_class> // FIXME Registerclass, dagoperand?
{
  def : Pat<(result_type (vop
                          (result_type result_reg_class:$rs3),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2))),
            (!cast<Instruction>(instruction_name#"_VV_M"#vlmul)
             result_reg_class:$rs3,
             // Floating point instructions with a scalar operand expect such
             // operand to be in a register of class FPR64. When dealing with
             // the f32 variant of such instructions we need to promote the
             // FPR32 subregister to the FPR64 base register
             !if(!eq(!cast<string>(op1_reg_class),
                     !cast<string>(FPR32)),
                 (SUBREG_TO_REG (i32 -1), FPR32:$rs1, sub_32),
                 (op1_type op1_reg_class:$rs1)),
             op2_reg_class:$rs2,
             X0,
             sew)>;
}

multiclass pat_vop_ternary_common<SDNode vop,
                                  string instruction_name,
                                  list<VectorTypeInfo> vtilist>
{
  foreach vti = vtilist in
  {
    defm : pat_vop_ternary<vop, instruction_name,
                           vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                           vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                           vti.RegClass, vti.RegClass>;
  }
}

let Predicates = [HasExtEPI] in {

defm "" : pat_vop_binary_common<add, "PseudoVADD", AllIntegerVectors>;
defm "" : pat_vop_binary_common<sub, "PseudoVSUB", AllIntegerVectors>;
defm "" : pat_vop_binary_common<mul, "PseudoVMUL", AllIntegerVectors>;

defm "" : pat_vop_binary_common<and, "PseudoVAND", AllIntegerVectors>;

defm "" : pat_vop_binary_common<fadd, "PseudoVFADD", AllFloatVectors>;
defm "" : pat_vop_binary_common<fsub, "PseudoVFSUB", AllFloatVectors>;
defm "" : pat_vop_binary_common<fmul, "PseudoVFMUL", AllFloatVectors>;

defm "" : pat_vop_ternary_common<fma, "PseudoVFMADD", AllFloatVectors>;

foreach fvti = AllFloatVectors in
{
  def : Pat<(fvti.Vector (riscv_vbroadcast fvti.ScalarRegClass:$value)),
            (!cast<Instruction>("PseudoVFMV_V_F_M"#fvti.RegClass.VLMul)
              !if(!eq(!cast<string>(fvti.ScalarRegClass),
                      !cast<string>(FPR32)),
                (SUBREG_TO_REG (i32 -1), FPR32:$value, sub_32),
                (fvti.Scalar fvti.ScalarRegClass:$value)),
              X0, fvti.SEW)>;

  foreach ivti = [GetIntVectorTypeInfo<fvti>.Vti] in 
  def : Pat<(fvti.Vector (sint_to_fp (ivti.Vector ivti.RegClass:$value))),
            (!cast<Instruction>("PseudoVFCVT_F_X_V_M"#fvti.RegClass.VLMul)
              $value, X0, fvti.SEW)>;

  foreach ivti = [GetIntVectorTypeInfo<fvti>.Vti] in 
  def : Pat<(fvti.Vector (uint_to_fp (ivti.Vector ivti.RegClass:$value))),
            (!cast<Instruction>("PseudoVFCVT_F_XU_V_M"#fvti.RegClass.VLMul)
              $value, X0, fvti.SEW)>;

  foreach ivti = [GetIntVectorTypeInfo<fvti>.Vti] in 
  def : Pat<(ivti.Vector (fp_to_sint (fvti.Vector fvti.RegClass:$value))),
            (!cast<Instruction>("PseudoVFCVT_X_F_V_M"#fvti.RegClass.VLMul)
              $value, X0, fvti.SEW)>;

  foreach ivti = [GetIntVectorTypeInfo<fvti>.Vti] in 
  def : Pat<(ivti.Vector (fp_to_uint (fvti.Vector fvti.RegClass:$value))),
            (!cast<Instruction>("PseudoVFCVT_XU_F_V_M"#fvti.RegClass.VLMul)
              $value, X0, fvti.SEW)>;
}

foreach vti = AllIntegerVectors in
{
  def : Pat<(vti.Vector (riscv_vbroadcast GPR:$value)),
            (!cast<Instruction>("PseudoVMV_V_X_M"#vti.RegClass.VLMul)
              GPR:$value,
              X0, vti.SEW)>;
  def : Pat<(vti.Vector (riscv_vbroadcast simm5:$value)),
            (!cast<Instruction>("PseudoVMV_V_I_M"#vti.RegClass.VLMul)
              simm5:$value,
              X0, vti.SEW)>;

  def : Pat<(vti.Vector (int_experimental_vector_stepvector)),
            (!cast<Instruction>("PseudoVID_V_M"#vti.RegClass.VLMul)
            X0, vti.SEW)>;
}

def : Pat<(extractelt (nxv1i64 EPIVR:$rs2), GPR:$rs1),
          (PseudoVEXT_X_V (nxv1i64 EPIVR:$rs2), GPR:$rs1, 64)>;

def : Pat<(extractelt (nxv2i64 EPIVR2:$rs2), GPR:$rs1),
          (PseudoVEXT_X_V
            (EXTRACT_SUBREG
              (PseudoVSLIDEDOWN_VX_M2 (nxv2i64 EPIVR2:$rs2), GPR:$rs1, X0, 64),
              epivreven),
            X0, 64)>;

def : Pat<(extractelt (nxv4i64 EPIVR4:$rs2), GPR:$rs1),
          (PseudoVEXT_X_V
            (EXTRACT_SUBREG
              (PseudoVSLIDEDOWN_VX_M4 (nxv4i64 EPIVR4:$rs2), GPR:$rs1, X0, 64),
              epivreven),
            X0, 64)>;

def : Pat<(extractelt (nxv8i64 EPIVR8:$rs2), GPR:$rs1),
          (PseudoVEXT_X_V
            (EXTRACT_SUBREG
              (PseudoVSLIDEDOWN_VX_M8 (nxv8i64 EPIVR8:$rs2), GPR:$rs1, X0, 64),
              epivreven),
            X0, 64)>;

// def : Pat<(sext_inreg (extractelt (nxv2i32 EPIVR:$rs2), GPR:$rs1), i32),
//           (PseudoVEXT_X_V $rs2, $rs1, 64)>;
// def : Pat<(sext_inreg (extractelt (nxv4i16 EPIVR:$rs2), GPR:$rs1), i16),
//           (PseudoVEXT_X_V $rs2, $rs1, 64)>;
// def : Pat<(sext_inreg (extractelt (nxv8i16 EPIVR:$rs2), GPR:$rs1), i8),
//           (PseudoVEXT_X_V $rs2, $rs1, 64)>;

// FIXME: This should be done in a better way.
def : Pat<(nxv1i64 (sext_inreg (nxv1i64 EPIVR:$value), nxv1i32)),
          (PseudoVSRA_VI_M1 (PseudoVSLL_VI_M1 $value, 32, X0, 64), 32, X0, 64)>;

}

//===----------------------------------------------------------------------===//
// Patterns. Load/store
//===----------------------------------------------------------------------===//

multiclass pat_load<string width> {
  foreach vti = AllVectors in
  foreach ivti = [GetIntVectorTypeInfo<vti>.Vti] in
  foreach vlmul = [vti.RegClass.VLMul] in
  {
    def : Pat<(vti.Vector (int_epi_vload GPR:$rs1, GPR:$vl)),
              (!cast<Instruction>("PseudoVL"#width#"_V_M"#vlmul) $rs1,
               (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_strided GPR:$rs1, GPR:$rs2, GPR:$vl)),
              (!cast<Instruction>("PseudoVLS"#width#"_V_M"#vlmul) $rs1, $rs2,
               (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_indexed GPR:$rs1,
                           (ivti.Vector vti.RegClass:$rs2), GPR:$vl)),
              (!cast<Instruction>("PseudoVLX"#width#"_V_M"#vlmul) $rs1, $rs2,
               (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_mask (vti.Vector vti.RegClass:$merge),
                           GPR:$rs1, (vti.Mask V0), GPR:$vl)),
               (!cast<Instruction>("PseudoVL"#width#"_V_MASK_M"#vlmul) $merge,
                $rs1, (vti.Mask V0), (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_strided_mask
                           (vti.Vector vti.RegClass:$merge), GPR:$rs1, GPR:$rs2,
                           (vti.Mask V0), GPR:$vl)),
              (!cast<Instruction>("PseudoVLS"#width#"_V_MASK_M"#vlmul) $merge,
               $rs1, $rs2, (vti.Mask V0), (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_indexed_mask
                           (vti.Vector vti.RegClass:$merge), GPR:$rs1,
                           (ivti.Vector vti.RegClass:$rs2), (vti.Mask V0),
                           GPR:$vl)),
              (!cast<Instruction>("PseudoVLX"#width#"_V_MASK_M"#vlmul) $merge,
               $rs1, $rs2, (vti.Mask V0), (NoX0 GPR:$vl), vti.SEW)>;
  }
}


multiclass pat_store<string width> {
  foreach vti = AllVectors in
  foreach ivti = [GetIntVectorTypeInfo<vti>.Vti] in
  foreach vlmul = [vti.RegClass.VLMul] in
  {
    def : Pat<(int_epi_vstore (vti.Vector vti.RegClass:$rs3), GPR:$rs1,
               GPR:$vl),
              (!cast<Instruction>("PseudoVS"#width#"_V_M"#vlmul)
               vti.RegClass:$rs3, GPR:$rs1, (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(int_epi_vstore_strided
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1, GPR:$rs2, GPR:$vl),
              (!cast<Instruction>("PseudoVSS"#width#"_V_M"#vlmul) $rs3, $rs1,
               $rs2, (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(int_epi_vstore_indexed
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1,
               (ivti.Vector vti.RegClass:$rs2), GPR:$vl),
              (!cast<Instruction>("PseudoVSX"#width#"_V_M"#vlmul) $rs3, $rs1,
               $rs2, (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(int_epi_vstore_mask
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1, (vti.Mask V0),
               GPR:$vl),
              (!cast<Instruction>("PseudoVS"#width#"_V_MASK_M"#vlmul) $rs3,
               $rs1, (vti.Mask V0), (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(int_epi_vstore_strided_mask
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1, GPR:$rs2,
               (vti.Mask V0), GPR:$vl),
              (!cast<Instruction>("PseudoVSS"#width#"_V_MASK_M"#vlmul) $rs3,
               $rs1, $rs2, (vti.Mask V0), (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(int_epi_vstore_indexed_mask
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1,
               (ivti.Vector vti.RegClass:$rs2), (vti.Mask V0), GPR:$vl),
              (!cast<Instruction>("PseudoVSX"#width#"_V_MASK_M"#vlmul) $rs3,
               $rs1, $rs2, (vti.Mask V0), (NoX0 GPR:$vl),
               vti.SEW)>;
  }
}

let Predicates = [HasExtEPI] in {

defm "" : pat_load<"E">;
defm "" : pat_store<"E">;

}
