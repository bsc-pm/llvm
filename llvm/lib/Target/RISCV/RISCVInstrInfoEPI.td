//=- RISCVInstrInfoV.td - Zeou-Extension RISCV instructions -*- tblgen-*----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

include "RISCVInstrFormatsEPI.td"

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

def vectormask : Operand<XLenVT> {
  let ParserMatchClass = VectorMaskOp;
  let PrintMethod = "printVectorMask";
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
      Uses = [VL, VTYPE, V0] in
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
      Uses = [VL, VTYPE, V0] in
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
      Uses = [VL, VTYPE, V0] in
  def I_MASK : RVInstEPIOPV_IVI<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, simm5:$imm5, vectormask:$vm),
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
      Uses = [VL, VTYPE, V0] in
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

multiclass binary_arithmetic_integer_vx<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_integer_n_v<opcodestr, funct6, "v">;
  defm _V : binary_arithmetic_integer_n_x<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_integer_xi<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_integer_n_x<opcodestr, funct6, "v">;
  defm _V : binary_arithmetic_integer_n_i<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_integer_v<string opcodestr, bits<6> funct6> {
  defm _V : binary_arithmetic_integer_n_v<opcodestr, funct6, "v">;
}

multiclass binary_arithmetic_integer_vxi_mask_in<string opcodestr, bits<6> funct6> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE, V0] in
  def _VV : RVInstEPIOPV_IVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, EPIVR:$rs1),
             opcodestr # ".vv",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE, V0] in
  def _VX : RVInstEPIOPV_IVX<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, GPR:$rs1),
             opcodestr # ".vx",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE, V0] in
  def _VI : RVInstEPIOPV_IVI<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, simm5:$imm5),
             opcodestr # ".vi",
             "${rd}, ${rs2}, ${imm5}">;
}

multiclass binary_arithmetic_integer_vxi_mask<string opcodestr, bits<6> funct6> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE, V0] in
  def V_MASK : RVInstEPIOPV_IVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, EPIVR:$rs1, vectormask:$vm),
             opcodestr # ".vv",
             "${rd}, ${rs2}, ${rs1}, ${vm}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE, V0] in
  def X_MASK : RVInstEPIOPV_IVX<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, GPR:$rs1, vectormask:$vm),
             opcodestr # ".vx",
             "${rd}, ${rs2}, ${rs1}, ${vm}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE, V0] in
  def I_MASK : RVInstEPIOPV_IVI<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, simm5:$imm5, vectormask:$vm),
             opcodestr # ".vi",
             "${rd}, ${rs2}, ${imm5}, ${vm}">;
}

multiclass binary_arithmetic_integer_vx_mask_in<string opcodestr, bits<6> funct6> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE, V0] in
  def _VV : RVInstEPIOPV_IVV<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, EPIVR:$rs1),
             opcodestr # ".vv",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Uses = [VL, VTYPE, V0] in
  def _VX : RVInstEPIOPV_IVX<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, GPR:$rs1),
             opcodestr # ".vx",
             "${rd}, ${rs2}, ${rs1}">;
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
      Uses = [VL, VTYPE, V0] in
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
      Uses = [VL, VTYPE, V0] in
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
             (ins EPIVR:$rs2, EPIFPR:$rs1),
             opcodestr # "." # destoperandkind # "f",
             "${rd}, ${rs2}, ${rs1}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Uses = [VL, VTYPE, V0] in
  def F_MASK : RVInstEPIOPV_FVF<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$rs2, EPIFPR:$rs1, vectormask:$vm),
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
      Constraints = "$rd = $_rd",Uses = [VL, VTYPE, V0] in
  def _VV_MASK : RVInstEPIOPV_FVV<funct6,
               (outs EPIVR:$rd),
               (ins EPIVR:$_rd, EPIVR:$rs1, EPIVR:$rs2, vectormask:$vm),
               opcodestr # ".vv",
               "${rd}, ${rs1}, ${rs2}, ${vm}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
      Constraints = "$rd = $_rd",Uses = [VL, VTYPE] in
  def _VF : RVInstEPIOPV_FVF<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$_rd, EPIFPR:$rs1, EPIVR:$rs2),
             opcodestr # ".vf",
             "${rd}, ${rs1}, ${rs2}">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
      Constraints = "$rd = $_rd",Uses = [VL, VTYPE, V0] in
  def _VF_MASK : RVInstEPIOPV_FVF<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$_rd, EPIFPR:$rs1, EPIVR:$rs2, vectormask:$vm),
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
      Uses = [VL, VTYPE, V0] in
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
      Uses = [VL, VTYPE, V0] in
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
      Uses = [VL, VTYPE, V0] in
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
      Constraints = "$rd = $_rd", Uses = [VL, VTYPE, V0] in
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
      Constraints = "$rd = $_rd", Uses = [VL, VTYPE, V0] in
  def _VX_MASK : RVInstEPIOPV_MVX<funct6,
             (outs EPIVR:$rd),
             (ins EPIVR:$_rd, GPR:$rs1, EPIVR:$rs2, vectormask:$vm),
             opcodestr # "." # destoperandkind # "x",
             "${rd}, ${rs1}, ${rs2}, ${vm}">;
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
      Uses = [VL, VTYPE, V0], rs1 = 0b00000 in
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
      Uses = [VL, VTYPE, V0], rs1 = rs1 in
  def _V_MASK : RVInstEPIOPV_FVV<funct6,
               (outs EPIVR:$rd),
               (ins EPIVR:$rs2, vectormask:$vm),
               opcodestr # ".v",
               "${rd}, ${rs2}, ${vm}">;
}

multiclass vector_store_unit_stride_width<string opcodestr, bits<3> width>
{
  let nf=0b000, rs2 = 0b00000, width = width,
      mayStore = 1, mayLoad = 0, hasSideEffects = 0 in {
    let vm = vmask_all_lanes.Value in
    def _V : RVInstStore<0b000,
               (outs),
               (ins EPIVR:$rd, GPR:$rs1),
               opcodestr # ".v",
               "${rd}, (${rs1})">;

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
      mayStore = 1, mayLoad = 0, hasSideEffects = 0 in {
    let vm = vmask_all_lanes.Value in
    def _V : RVInstStore<mop,
               (outs),
               (ins EPIVR:$rd, GPR:$rs1, rc:$rs2),
               opcodestr # ".v",
               "${rd}, (${rs1}), ${rs2}">;

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
      mayStore = 0, mayLoad = 1, hasSideEffects = 0 in {
    let vm = vmask_all_lanes.Value in
    def _V : RVInstLoad<{sign, 0b00},
               (outs EPIVR:$rd),
               (ins GPR:$rs1),
               opcodestr # ".v",
               "${rd}, (${rs1})">;

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
      mayStore = 0, mayLoad = 1, hasSideEffects = 0 in {
    let vm = vmask_all_lanes.Value in
    def _V : RVInstLoad<{sign, mop},
               (outs EPIVR:$rd),
               (ins GPR:$rs1, rc:$rs2),
               opcodestr # ".v",
               "${rd}, (${rs1}), ${rs2}">;

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

defm VSLIDEUP   : binary_arithmetic_integer_xi<"vslideup", 0b001110>;
defm VSLIDEDOWN : binary_arithmetic_integer_xi<"vslidedown", 0b001111>;

defm VADC : binary_arithmetic_integer_vxi_mask_in<"vadc", 0b010000>;
defm VSBC : binary_arithmetic_integer_vx_mask_in<"vsbc", 0b010010>;

defm VMERGE : binary_arithmetic_integer_vxi<"vmerge", 0b010111>;

defm VSEQ   : binary_arithmetic_integer_vxi<"vseq",  0b011000>;
defm VSNE   : binary_arithmetic_integer_vxi<"vsne",  0b011001>;
defm VSLTU  : binary_arithmetic_integer_vx<"vsltu",  0b011010>;
defm VSLT   : binary_arithmetic_integer_vx<"vslt",   0b011011>;
defm VSLEU  : binary_arithmetic_integer_vxi<"vsleu", 0b011100>;
defm VSLE   : binary_arithmetic_integer_vxi<"vsle",  0b011101>;
defm VSGTU  : binary_arithmetic_integer_xi<"vsgtu",  0b011110>;
defm VSGT   : binary_arithmetic_integer_xi<"vsgt",   0b011111>;

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
    Uses = [VL, VTYPE, V0], rs1 = 0b00001 in
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
    Uses = [VL, VTYPE, V0], rs1 = 0b00010 in
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
    Uses = [VL, VTYPE, V0], rs1 = 0b00011 in
def VMSIF_M_MASK : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins EPIVR:$rs2, vectormask:$vm),
  "vmsif.m", "${rd}, ${rs2}, ${vm}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE], rs1 = 0b10000 in
def VMIOTA_M : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins EPIVR:$rs2),
  "vmiota.m", "${rd}, ${rs2}">;
let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
    Uses = [VL, VTYPE, V0], rs1 = 0b10000 in
def VMIOTA_M_MASK : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins EPIVR:$rs2, vectormask:$vm),
  "vmiota.m", "${rd}, ${rs2}, ${vm}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE], rs1 = 0b10001, rs2 = 0b00000 in
def VID_V : RVInstEPIOPV_MVV<0b010110,
  (outs EPIVR:$rd),
  (ins),
  "vid.v", "${rd}">;
let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
    Uses = [VL, VTYPE, V0], rs1 = 0b10001, rs2 = 0b00000 in
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
           (outs EPIFPR:$rd),
           (ins EPIVR:$rs2),
           "vfmv.f.s", "${rd}, ${rs2}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE], rs2 = 0b00000 in
def VFMV_S_F : RVInstEPIOPV_FVF<0b001101,
           (outs EPIVR:$rd),
           (ins EPIFPR:$rs1),
           "vfmv.s.f", "${rd}, ${rs1}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, vm = vmask_all_lanes.Value,
    Uses = [VL, VTYPE] in
def VFMERGE_VF : RVInstEPIOPV_FVF<0b010111,
           (outs EPIVR:$rd),
           (ins EPIVR:$rs2, EPIFPR:$rs1),
           "vfmerge.vf",
           "${rd}, ${rs2}, ${rs1}">;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
    Uses = [VL, VTYPE, V0] in
def VFMERGE_VF_MASK : RVInstEPIOPV_FVF<0b010111,
           (outs EPIVR:$rd),
           (ins EPIVR:$rs2, EPIFPR:$rs1, vectormask:$vm),
           "vfmerge.vf",
           "${rd}, ${rs2}, ${rs1}, ${vm}">;

defm VFEQ    : binary_arithmetic_float_vf<"vfeq",  0b011000>;
defm VFLTE   : binary_arithmetic_float_vf<"vflte", 0b011001>;
defm VFORD   : binary_arithmetic_float_vf<"vford", 0b011010>;
defm VFLT    : binary_arithmetic_float_vf<"vflt",  0b011011>;
defm VFNE    : binary_arithmetic_float_vf<"vfne",  0b011100>;
defm VFGT    : binary_arithmetic_float_f<"vfgt",  0b011101>;
defm VFGTE   : binary_arithmetic_float_f<"vfgte", 0b011111>;

defm VFDIV   : binary_arithmetic_float_vf<"vfdiv",  0b100000>;
defm VFRDIV  : binary_arithmetic_float_vf<"vfrdiv", 0b100001>;

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

defm VFSQRT : unary_arithmetic_float_v<"vfsqrt", 0b100011, 0b00000>;
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
                          (ins), [], "readvl $rd">;

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
// Patterns
//===----------------------------------------------------------------------===//

def EPIIntrinsicsTable : GenericTable {
  let FilterClass = "EPIIntrinsic";
  let CppTypeName = "EPIIntrinsicInfo";
  let Fields = [ "IntrinsicID", "ExtendOperand" ];
  let PrimaryKey = [ "IntrinsicID" ];
  let PrimaryKeyName = "getEPIIntrinsicInfo";
}

//===----------------------------------------------------------------------===//
// Patterns. Essential
//===----------------------------------------------------------------------===//

// TODO: Complete patterns
multiclass pat_load_store<LLVMType ty, RVInst load_inst, RVInst store_inst> {

  // Load
  def : Pat<(ty (load GPR:$rs1)),
            (load_inst GPR:$rs1)>;
  def : Pat<(ty (load AddrFI:$rs1)),
            (load_inst AddrFI:$rs1)>;

  // Store
  def : Pat<(store ty:$rs2, GPR:$rs1),
            (store_inst EPIVR:$rs2, GPR:$rs1)>;
  def : Pat<(store ty:$rs2, AddrFI:$rs1),
            (store_inst EPIVR:$rs2, AddrFI:$rs1)>;
}

let Predicates = [HasExtEPI] in {

  defm : pat_load_store<nxv1i1,  VLB_V, VSB_V>;
  defm : pat_load_store<nxv1i8,  VLB_V, VSB_V>;
  defm : pat_load_store<nxv1i16, VLH_V, VSH_V>;
  defm : pat_load_store<nxv1i32, VLW_V, VSW_V>;
  defm : pat_load_store<nxv1i64, VLE_V, VSE_V>;

  defm : pat_load_store<nxv1f32, VLW_V, VSW_V>;
  defm : pat_load_store<nxv1f64, VLE_V, VSE_V>;
}

//===----------------------------------------------------------------------===//
// Patterns. Vector register configuration
//===----------------------------------------------------------------------===//

let Predicates = [HasExtEPI] in {

// FIXME: I think we better map this one to VLEN (i.e. the maximum vector
// length in bits)
def : Pat<(int_experimental_vector_vscale), (PseudoReadVL)>;

def : Pat<(int_epi_readvl), (PseudoReadVL)>;

// FIXME: uimm5 here is not very precise. It should be uimm3 and uimm2 but we
// will probably want to wait for intrinsics with constant values constraints.
def : Pat<(int_epi_setvl GPR:$vl, uimm5:$sew, uimm5:$vmul),
          (VSETVLI GPR:$vl, uimm5:$sew, uimm5:$vmul)>;

}

//===----------------------------------------------------------------------===//
// Patterns. Arithmetic
//===----------------------------------------------------------------------===//

class VectorScalar<ValueType Vec, ValueType Scal> {
   ValueType Vector = Vec;
   ValueType Scalar = Scal;
}

class ValueTypeList<list<VectorScalar> Vs>
{
  list<VectorScalar> Values = Vs;
}

def AllIntegerVectors
    : ValueTypeList<[VectorScalar<nxv1i8, i8>,
                     VectorScalar<nxv1i16, i16>,
                     VectorScalar<nxv1i32, i32>,
                     VectorScalar<nxv1i64, i64>]>;

def AllFloatVectors
    : ValueTypeList<[VectorScalar<nxv1f32, f32>,
                     VectorScalar<nxv1f64, f64>]>;

def AllVectors
   : ValueTypeList<!listconcat(AllIntegerVectors.Values,
                               AllFloatVectors.Values)>;


multiclass pat_intrinsic_binary_vv_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), (op_type EPIVR:$rs2))),
            (!cast<Instruction>(instruction_name # "_VV")
               EPIVR:$rs1, EPIVR:$rs2)>;
}

multiclass pat_intrinsic_binary_vv<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  defm : pat_intrinsic_binary_vv_nomask<intrinsic_name, instruction_name,
                                           op_type, result_type>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                         (op_type EPIVR:$rs1),
                         (op_type EPIVR:$rs2), V0)),
            (!cast<Instruction>(instruction_name # "_VV_MASK")
               EPIVR:$rs1, EPIVR:$rs2, vmask_only_true.Value)>;
}

multiclass pat_intrinsic_binary_vs_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), (op_type EPIVR:$rs2))),
            (!cast<Instruction>(instruction_name # "_VS")
               EPIVR:$rs1, EPIVR:$rs2)>;
}

multiclass pat_intrinsic_binary_vs<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  defm : pat_intrinsic_binary_vs_nomask<intrinsic_name, instruction_name,
                                           op_type, result_type>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                         (op_type EPIVR:$rs1),
                         (op_type EPIVR:$rs2), V0)),
            (!cast<Instruction>(instruction_name # "_VS_MASK")
               EPIVR:$rs1, EPIVR:$rs2, vmask_only_true.Value)>;
}

multiclass pat_intrinsic_binary_vx_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), GPR:$rs2)),
            (!cast<Instruction>(instruction_name # "_VX")
               EPIVR:$rs1, GPR:$rs2)>;
}

multiclass pat_intrinsic_binary_vx<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  defm : pat_intrinsic_binary_vx_nomask<intrinsic_name, instruction_name,
                                           op_type, result_type>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                         (op_type EPIVR:$rs1),
                         GPR:$rs2, V0)),
            (!cast<Instruction>(instruction_name # "_VX_MASK")
               EPIVR:$rs1, GPR:$rs2, vmask_only_true.Value)>;
}

multiclass pat_intrinsic_binary_vi_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), simm5:$imm5)),
            (!cast<Instruction>(instruction_name # "_VI")
               EPIVR:$rs1, simm5:$imm5)>;
}

multiclass pat_intrinsic_binary_vi<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  defm "" : pat_intrinsic_binary_vi_nomask<intrinsic_name, instruction_name, 
                                           op_type, result_type>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                         (op_type EPIVR:$rs1),
                         (simm5:$imm5), V0)),
            (!cast<Instruction>(instruction_name # "_VI_MASK")
               EPIVR:$rs1, simm5:$imm5, vmask_only_true.Value)>;
}

multiclass pat_intrinsic_binary_vf_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType op_scalar,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), (op_scalar EPIFPR:$rs2))),
            (!cast<Instruction>(instruction_name # "_VF")
               EPIVR:$rs1, EPIFPR:$rs2)>;
}

multiclass pat_intrinsic_binary_vf<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType op_scalar,
                                   ValueType result_type>
{
  defm : pat_intrinsic_binary_vf_nomask<intrinsic_name,
                   instruction_name, op_type, op_scalar, result_type>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                         (op_type EPIVR:$rs1),
                         (op_scalar EPIFPR:$rs2), V0)),
            (!cast<Instruction>(instruction_name # "_VF_MASK")
               EPIVR:$rs1, EPIFPR:$rs2, vmask_only_true.Value)>;
}

multiclass pat_intrinsic_binary_int_vv_vx_vi<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_vv<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
    defm : pat_intrinsic_binary_vx<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
    defm : pat_intrinsic_binary_vi<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
  }
}


multiclass pat_intrinsic_binary_int_vv_vx<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_vv<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
    defm : pat_intrinsic_binary_vx<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_int_vv<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_vv<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_int_vs<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_vs<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_int_vv_nomask<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_vv_nomask<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_int_vx<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_vx<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_int_vx_vi<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_vx<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
    defm : pat_intrinsic_binary_vi<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_relational_int_vv_vx_vi<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_vv<intrinsic_name, instruction_name, vtp.Vector, nxv1i1>;
    defm : pat_intrinsic_binary_vx<intrinsic_name, instruction_name, vtp.Vector, nxv1i1>;
    defm : pat_intrinsic_binary_vi<intrinsic_name, instruction_name, vtp.Vector, nxv1i1>;
  }
}

multiclass pat_intrinsic_binary_relational_int_vv_vx<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_vv<intrinsic_name, instruction_name, vtp.Vector, nxv1i1>;
    defm : pat_intrinsic_binary_vx<intrinsic_name, instruction_name, vtp.Vector, nxv1i1>;
  }
}

multiclass pat_intrinsic_binary_relational_int_vv_vi<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_vv<intrinsic_name, instruction_name, vtp.Vector, nxv1i1>;
    defm : pat_intrinsic_binary_vi<intrinsic_name, instruction_name, vtp.Vector, nxv1i1>;
  }
}

multiclass pat_intrinsic_binary_relational_int_vx_vi<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_vx<intrinsic_name, instruction_name, vtp.Vector, nxv1i1>;
    defm : pat_intrinsic_binary_vi<intrinsic_name, instruction_name, vtp.Vector, nxv1i1>;
  }
}

multiclass pat_intrinsic_binary_with_mask_input_vv_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), (op_type EPIVR:$rs2), (op_type V0))),
            (!cast<Instruction>(instruction_name # "_VV")
               EPIVR:$rs1, EPIVR:$rs2)>;
}

multiclass pat_intrinsic_binary_with_mask_input_vx_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), GPR:$rs2, (op_type V0))),
            (!cast<Instruction>(instruction_name # "_VX")
               EPIVR:$rs1, GPR:$rs2)>;
}

multiclass pat_intrinsic_binary_with_mask_input_vi_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), simm5:$imm5, (op_type V0))),
            (!cast<Instruction>(instruction_name # "_VI")
               EPIVR:$rs1, simm5:$imm5)>;
}

multiclass pat_intrinsic_binary_with_mask_input_int_vv_vx_vi_nomask<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_with_mask_input_vv_nomask<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
    defm : pat_intrinsic_binary_with_mask_input_vx_nomask<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
    defm : pat_intrinsic_binary_with_mask_input_vi_nomask<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_with_mask_input_int_vv_vx_nomask<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_with_mask_input_vv_nomask<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
    defm : pat_intrinsic_binary_with_mask_input_vx_nomask<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_fp_vv<string intrinsic_name,
                                      string instruction_name>
{
  foreach vtp = AllFloatVectors.Values
  in {
    defm : pat_intrinsic_binary_vv<intrinsic_name, instruction_name,
                                          vtp.Vector, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_fp_vf<string intrinsic_name,
                                      string instruction_name>
{
  foreach vtp = AllFloatVectors.Values
  in {
    defm : pat_intrinsic_binary_vf<intrinsic_name, instruction_name,
                                          vtp.Vector, vtp.Scalar, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_fp_vs<string intrinsic_name,
                                      string instruction_name>
{
  foreach vtp = AllFloatVectors.Values
  in {
    defm : pat_intrinsic_binary_vs<intrinsic_name, instruction_name,
                                          vtp.Vector, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_fp_vv_vf<string intrinsic_name,
                                         string instruction_name>
{
  foreach vtp = AllFloatVectors.Values
  in {
    defm : pat_intrinsic_binary_vv<intrinsic_name, instruction_name,
                                          vtp.Vector, vtp.Vector>;
    defm : pat_intrinsic_binary_vf<intrinsic_name, instruction_name,
                                          vtp.Vector, vtp.Scalar, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_relational_fp_vv_vf<string intrinsic_name,
                                         string instruction_name>
{
  foreach vtp = AllFloatVectors.Values
  in {
    defm : pat_intrinsic_binary_vv<intrinsic_name, instruction_name,
                                          vtp.Vector, nxv1i1>;
    defm : pat_intrinsic_binary_vf<intrinsic_name, instruction_name,
                                          vtp.Vector, vtp.Scalar, nxv1i1>;
  }
}

multiclass pat_intrinsic_binary_relational_fp_vf<string intrinsic_name,
                                         string instruction_name>
{
  foreach vtp = AllFloatVectors.Values
  in {
    defm : pat_intrinsic_binary_vf<intrinsic_name, instruction_name,
                                          vtp.Vector, vtp.Scalar, nxv1i1>;
  }
}

multiclass pat_intrinsic_binary_int_mm<string intrinsic_name,
                                       string instruction_name>
{
  def : Pat<(nxv1i1 (!cast<Intrinsic>(intrinsic_name)
                         (nxv1i1 EPIVR:$rs1),
                         (nxv1i1 EPIVR:$rs2))),
            (!cast<Instruction>(instruction_name # "_MM")
               EPIVR:$rs1, EPIVR:$rs2)>;
}

multiclass pat_intrinsic_ternary_vv_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), (op_type EPIVR:$rs2), (op_type EPIVR:$rs3))),
            (!cast<Instruction>(instruction_name # "_VV")
               EPIVR:$rs3, EPIVR:$rs2, EPIVR:$rs1)>;
}

multiclass pat_intrinsic_ternary_vv<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  defm : pat_intrinsic_ternary_vv_nomask<intrinsic_name, instruction_name,
                                           op_type, result_type>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                         (op_type EPIVR:$rs1),
                         (op_type EPIVR:$rs2),
                         (op_type EPIVR:$rs3), V0)),
            (!cast<Instruction>(instruction_name # "_VV_MASK")
               EPIVR:$rs3, EPIVR:$rs2, EPIVR:$rs1, vmask_only_true.Value)>;
}

multiclass pat_intrinsic_ternary_vx_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
               (op_type EPIVR:$rs1),
               GPR:$rs2,
               (op_type EPIVR:$rs3))),
            (!cast<Instruction>(instruction_name # "_VX")
               EPIVR:$rs3, GPR:$rs2, EPIVR:$rs1)>;
}

multiclass pat_intrinsic_ternary_vx<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  defm : pat_intrinsic_ternary_vx_nomask<intrinsic_name, instruction_name,
                                           op_type, result_type>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                         (op_type EPIVR:$rs1),
                         GPR:$rs2,
                         (op_type EPIVR:$rs3),
                         V0)),
            (!cast<Instruction>(instruction_name # "_VX_MASK")
               EPIVR:$rs3, GPR:$rs2, EPIVR:$rs1, vmask_only_true.Value)>;
}

multiclass pat_intrinsic_ternary_int_vv_vx<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_ternary_vv<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
    defm : pat_intrinsic_ternary_vx<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
  }
}

multiclass pat_intrinsic_ternary_vf_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType op_scalar,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
               (op_type EPIVR:$rs1),
               (op_scalar EPIFPR:$rs2),
               (op_type EPIVR:$rs3))),
            (!cast<Instruction>(instruction_name # "_VF")
               EPIVR:$rs3, EPIFPR:$rs2, EPIVR:$rs1)>;
}

multiclass pat_intrinsic_ternary_vf<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType op_scalar,
                                   ValueType result_type>
{
  defm "" : pat_intrinsic_ternary_vf_nomask<intrinsic_name, instruction_name,
                                           op_type, op_scalar, result_type>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                         (op_type EPIVR:$rs1),
                         (op_scalar EPIFPR:$rs2),
                         (op_type EPIVR:$rs3),
                         V0)),
            (!cast<Instruction>(instruction_name # "_VF_MASK")
               EPIVR:$rs3, EPIFPR:$rs2, EPIVR:$rs1, vmask_only_true.Value)>;
}

multiclass pat_intrinsic_ternary_fp_vv_vf<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllFloatVectors.Values
  in {
    defm : pat_intrinsic_ternary_vv<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
    defm : pat_intrinsic_ternary_vf<intrinsic_name, instruction_name, vtp.Vector, vtp.Scalar, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_wv_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), (op_type EPIVR:$rs2))),
            (!cast<Instruction>(instruction_name # "_WV")
               EPIVR:$rs1, EPIVR:$rs2)>;
}

multiclass pat_intrinsic_binary_wv<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  defm "" : pat_intrinsic_binary_wv_nomask<intrinsic_name, instruction_name,
                                           op_type, result_type>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                         (op_type EPIVR:$rs1),
                         (op_type EPIVR:$rs2), V0)),
            (!cast<Instruction>(instruction_name # "_WV_MASK")
               EPIVR:$rs1, EPIVR:$rs2, vmask_only_true.Value)>;
}

multiclass pat_intrinsic_binary_wf_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType op_scalar,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), (op_scalar EPIFPR:$rs2))),
            (!cast<Instruction>(instruction_name # "_WF")
               EPIVR:$rs1, EPIFPR:$rs2)>;
}

multiclass pat_intrinsic_binary_wf<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType op_scalar,
                                   ValueType result_type>
{
  defm "" : pat_intrinsic_binary_wf_nomask<intrinsic_name,
                   instruction_name, op_type, op_scalar, result_type>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                         (op_type EPIVR:$rs1),
                         (op_scalar EPIFPR:$rs2), V0)),
            (!cast<Instruction>(instruction_name # "_WF_MASK")
               EPIVR:$rs1, EPIFPR:$rs2, vmask_only_true.Value)>;
}

multiclass pat_intrinsic_binary_fp_wv_wf<string intrinsic_name,
                                         string instruction_name>
{
  foreach vtp = AllFloatVectors.Values
  in {
    defm : pat_intrinsic_binary_wv<intrinsic_name, instruction_name,
                                          vtp.Vector, vtp.Vector>;
    defm : pat_intrinsic_binary_wf<intrinsic_name, instruction_name,
                                          vtp.Vector, vtp.Scalar, vtp.Vector>;
  }
}


multiclass pat_intrinsic_binary_wx_nomask<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type EPIVR:$rs1), GPR:$rs2)),
            (!cast<Instruction>(instruction_name # "_WX")
               EPIVR:$rs1, GPR:$rs2)>;
}

multiclass pat_intrinsic_binary_wx<string intrinsic_name,
                                   string instruction_name,
                                   ValueType op_type,
                                   ValueType result_type>
{
  defm "" : pat_intrinsic_binary_wx_nomask<intrinsic_name, instruction_name,
                                           op_type, result_type>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                         (op_type EPIVR:$rs1),
                         GPR:$rs2, V0)),
            (!cast<Instruction>(instruction_name # "_WX_MASK")
               EPIVR:$rs1, GPR:$rs2, vmask_only_true.Value)>;
}

multiclass pat_intrinsic_binary_int_wv_wx<string intrinsic_name,
                                    string instruction_name>
{
  foreach vtp = AllIntegerVectors.Values
  in {
    defm : pat_intrinsic_binary_wv<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
    defm : pat_intrinsic_binary_wx<intrinsic_name, instruction_name, vtp.Vector, vtp.Vector>;
  }
}

multiclass pat_intrinsic_binary_any_and_int_vv<string intrinsic_name,
                                    string instruction_name>
{
    foreach vtp = AllVectors.Values in
    {
        foreach itp = AllIntegerVectors.Values in
        {
            def : Pat<(vtp.Vector (!cast<Intrinsic>(intrinsic_name) (vtp.Vector EPIVR:$rs1), (itp.Vector EPIVR:$rs2))),
                      (!cast<Instruction>(instruction_name # "_VV")
                         EPIVR:$rs1, EPIVR:$rs2)>;
            def : Pat<(vtp.Vector (!cast<Intrinsic>(intrinsic_name # "_mask")
                                   (vtp.Vector EPIVR:$rs1),
                                   (itp.Vector EPIVR:$rs2), V0)),
                      (!cast<Instruction>(instruction_name # "_VV_MASK")
                         EPIVR:$rs1, EPIVR:$rs2, vmask_only_true.Value)>;
        }

   }
}

multiclass pat_intrinsic_binary_any_and_int_vx<string intrinsic_name,
                                    string instruction_name>
{
    foreach vtp = AllVectors.Values in
    {
        def : Pat<(vtp.Vector (!cast<Intrinsic>(intrinsic_name) (vtp.Vector EPIVR:$rs1), GPR:$rs2)),
                  (!cast<Instruction>(instruction_name # "_VX")
                     EPIVR:$rs1, GPR:$rs2)>;
        def : Pat<(vtp.Vector (!cast<Intrinsic>(intrinsic_name # "_mask")
                               (vtp.Vector EPIVR:$rs1),
                               GPR:$rs2, V0)),
                  (!cast<Instruction>(instruction_name # "_VX_MASK")
                     EPIVR:$rs1, GPR:$rs2, vmask_only_true.Value)>;
   }
}

multiclass pat_intrinsic_binary_any_and_int_vi<string intrinsic_name,
                                    string instruction_name>
{
    foreach vtp = AllVectors.Values in
    {
        def : Pat<(vtp.Vector (!cast<Intrinsic>(intrinsic_name) (vtp.Vector EPIVR:$rs1), (simm5:$imm5))),
                  (!cast<Instruction>(instruction_name # "_VI")
                     EPIVR:$rs1, simm5:$imm5)>;
        def : Pat<(vtp.Vector (!cast<Intrinsic>(intrinsic_name # "_mask")
                               (vtp.Vector EPIVR:$rs1),
                               simm5:$imm5, V0)),
                  (!cast<Instruction>(instruction_name # "_VI_MASK")
                     EPIVR:$rs1, simm5:$imm5, vmask_only_true.Value)>;
   }
}


multiclass pat_intrinsic_binary_any_and_int_vv_vx_vi<string intrinsic_name,
                                    string instruction_name>
{
  defm : pat_intrinsic_binary_any_and_int_vv<intrinsic_name, instruction_name>;
  defm : pat_intrinsic_binary_any_and_int_vx<intrinsic_name, instruction_name>;
  defm : pat_intrinsic_binary_any_and_int_vi<intrinsic_name, instruction_name>;
}

multiclass pat_intrinsic_binary_any_and_int_vx_vi<string intrinsic_name,
                                    string instruction_name>
{
  defm : pat_intrinsic_binary_any_and_int_vx<intrinsic_name, instruction_name>;
  defm : pat_intrinsic_binary_any_and_int_vi<intrinsic_name, instruction_name>;
}


class ValueTypePair<ValueType ty1, ValueType ty2>
{
   ValueType First = ty1;
   ValueType Second = ty2;
}


multiclass pat_conversions_symmetrical<string first_instruction,
                           string first_intrinsic,
                           string second_instruction,
                           string second_intrinsic,
                           list<ValueTypePair> pairs>
{

foreach vtp = pairs in
{
def : Pat<(vtp.First (!cast<Intrinsic>("int_epi_" # first_intrinsic) (vtp.Second EPIVR:$rs1))),
          (!cast<Instruction>(first_instruction # "_V") EPIVR:$rs1)>;
def : Pat<(vtp.First (!cast<Intrinsic>("int_epi_" # first_intrinsic # "_mask") (vtp.Second EPIVR:$rs1), V0)),
          (!cast<Instruction>(first_instruction # "_V_MASK") EPIVR:$rs1, vmask_only_true.Value)>;

def : Pat<(vtp.Second (!cast<Intrinsic>("int_epi_" # second_intrinsic) (vtp.First EPIVR:$rs1))),
          (!cast<Instruction>(second_instruction # "_V") EPIVR:$rs1)>;
def : Pat<(vtp.Second (!cast<Intrinsic>("int_epi_" # second_intrinsic # "_mask") (vtp.First EPIVR:$rs1), V0)),
          (!cast<Instruction>(second_instruction # "_V_MASK") EPIVR:$rs1, vmask_only_true.Value)>;

}

}

class ValueTypeVarList<list<ValueTypePair> VList>
{
  list<ValueTypePair> Value = VList;
}

def SameSizePairs : ValueTypeVarList<
  [ ValueTypePair<nxv1i32, nxv1f32>,
    ValueTypePair<nxv1i64, nxv1f64> ]>;

multiclass pat_conversions<string instruction,
                           string intrinsic,
                           list<ValueTypePair> pairs>
{

foreach vtp = pairs in
{
def : Pat<(vtp.First (!cast<Intrinsic>("int_epi_" # intrinsic) (vtp.Second EPIVR:$rs1))),
          (!cast<Instruction>(instruction # "_V") EPIVR:$rs1)>;
def : Pat<(vtp.First (!cast<Intrinsic>("int_epi_" # intrinsic # "_mask") (vtp.Second EPIVR:$rs1), V0)),
          (!cast<Instruction>(instruction # "_V_MASK") EPIVR:$rs1, vmask_only_true.Value)>;

}

}

// We are missing cases but are of sizes that won't work
def WidenedSize : ValueTypeVarList<
  [ ValueTypePair<nxv1i64, nxv1f32>,
    ValueTypePair<nxv1f32, nxv1i16>,
    ValueTypePair<nxv1f64, nxv1i32> ]>;

// Narrowing
// TODO: This list is the inverse of the one above
def NarrowedSize : ValueTypeVarList<
  [ ValueTypePair<nxv1f32, nxv1i64>,
    ValueTypePair<nxv1i16, nxv1f32>,
    ValueTypePair<nxv1i32, nxv1f64> ]>;

def FloatWidenedSize : ValueTypeVarList<
  [ ValueTypePair<nxv1f64, nxv1f32> ]>;

def FloatNarrowedSize : ValueTypeVarList<
  [ ValueTypePair<nxv1f32, nxv1f64> ]>;

let Predicates = [HasExtEPI] in {

defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vadd", "VADD">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vsub", "VSUB">;
defm "" : pat_intrinsic_binary_int_vx_vi<"int_epi_vrsub", "VRSUB">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vminu", "VMINU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vmin", "VMIN">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vmaxu", "VMAXU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vmax", "VMAX">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vand", "VAND">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vor", "VOR">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vxor", "VXOR">;

// These pattern are missing floats
defm "" : pat_intrinsic_binary_any_and_int_vv_vx_vi<"int_epi_vrgather", "VRGATHER">;
defm "" : pat_intrinsic_binary_any_and_int_vx_vi<"int_epi_vslideup", "VSLIDEUP">;
defm "" : pat_intrinsic_binary_any_and_int_vx_vi<"int_epi_vslidedown", "VSLIDEDOWN">;

defm "" : pat_intrinsic_binary_with_mask_input_int_vv_vx_vi_nomask<"int_epi_vadc", "VADC">;
defm "" : pat_intrinsic_binary_with_mask_input_int_vv_vx_nomask<"int_epi_vsbc", "VSBC">;

defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vmerge", "VMERGE">;

defm "" : pat_intrinsic_binary_relational_int_vv_vx_vi<"int_epi_vseq", "VSEQ">;
defm "" : pat_intrinsic_binary_relational_int_vv_vx_vi<"int_epi_vsne", "VSNE">;
defm "" : pat_intrinsic_binary_relational_int_vv_vx<"int_epi_vsltu", "VSLTU">;
defm "" : pat_intrinsic_binary_relational_int_vv_vx<"int_epi_vslt", "VSLT">;
defm "" : pat_intrinsic_binary_relational_int_vv_vx_vi<"int_epi_vsleu", "VSLEU">;
defm "" : pat_intrinsic_binary_relational_int_vv_vx_vi<"int_epi_vsle", "VSLE">;
defm "" : pat_intrinsic_binary_relational_int_vx_vi<"int_epi_vsgtu", "VSGTU">;
defm "" : pat_intrinsic_binary_relational_int_vx_vi<"int_epi_vsgt", "VSGT">;

defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vsaddu", "VSADDU">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vsadd", "VSADD">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vssubu", "VSSUBU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vssub", "VSSUB">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vaadd", "VAADD">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vsll", "VSLL">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vasub", "VASUB">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vsmul", "VSMUL">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vsrl", "VSRL">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vsra", "VSRA">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vssrl", "VSSRL">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vssra", "VSSRA">;

defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vmerge", "VMERGE">;

// FIXME: These patterns are wrong
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vnsrl", "VNSRL">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vnsra", "VNSRA">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vnclipu", "VNCLIPU">;
defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vnclip", "VNCLIP">;

// FIXME: These patterns are wrong
defm "" : pat_intrinsic_binary_int_vs<"int_epi_vwredsumu", "VWREDSUMU">;
defm "" : pat_intrinsic_binary_int_vs<"int_epi_vwredsum", "VWREDSUM">;

defm "" : pat_intrinsic_binary_int_vv<"int_epi_vdotu", "VDOTU">;
defm "" : pat_intrinsic_binary_int_vv<"int_epi_vdot", "VDOT">;

// FIXME: These patterns are wrong
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmaccu", "VWSMACCU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmacc", "VWSMACC">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmsacu", "VWSMSACU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmsac", "VWSMSAC">;

defm "" : pat_intrinsic_binary_int_vs<"int_epi_vredsum", "VREDSUM">;
defm "" : pat_intrinsic_binary_int_vs<"int_epi_vredand", "VREDAND">;
defm "" : pat_intrinsic_binary_int_vs<"int_epi_vredor", "VREDOR">;
defm "" : pat_intrinsic_binary_int_vs<"int_epi_vredxor", "VREDXOR">;
defm "" : pat_intrinsic_binary_int_vs<"int_epi_vredminu", "VREDMINU">;
defm "" : pat_intrinsic_binary_int_vs<"int_epi_vredmin", "VREDMIN">;
defm "" : pat_intrinsic_binary_int_vs<"int_epi_vredmaxu", "VREDMAXU">;
defm "" : pat_intrinsic_binary_int_vs<"int_epi_vredmax", "VREDMAX">;

foreach vtp = AllIntegerVectors.Values in
def : Pat<(int_epi_vext_x_v (vtp.Vector EPIVR:$rs2), GPR:$rs1),
          (VEXT_X_V $rs2, $rs1)>;

foreach vtp = AllIntegerVectors.Values in
def : Pat<(vtp.Vector (int_epi_vmv_s_x GPR:$rs2)),
          (VMV_S_X $rs2)>;


defm "" : pat_intrinsic_binary_any_and_int_vx<"int_epi_vslide1up", "VSLIDE1UP">;
defm "" : pat_intrinsic_binary_any_and_int_vx<"int_epi_vslide1down", "VSLIDE1DOWN">;

foreach vtp = [nxv1i1] in {
def : Pat<(int_epi_vmpopc (vtp EPIVR:$rs2)),
          (VMPOPC_M $rs2)>;
def : Pat<(int_epi_vmpopc_mask (vtp EPIVR:$rs2), V0),
          (VMPOPC_M_MASK $rs2, vmask_only_true.Value)>;

def : Pat<(int_epi_vmfirst (vtp EPIVR:$rs2)),
          (VMFIRST_M $rs2)>;
def : Pat<(int_epi_vmfirst_mask (vtp EPIVR:$rs2), V0),
          (VMFIRST_M_MASK $rs2, vmask_only_true.Value)>;
}

def : Pat<(int_epi_vmsbf EPIVR:$rs1),
        (VMSBF_M $rs1)>;
def : Pat<(int_epi_vmsbf_mask EPIVR:$rs1, V0),
        (VMSBF_M_MASK $rs1, vmask_only_true.Value)>;

def : Pat<(int_epi_vmsif EPIVR:$rs1),
        (VMSIF_M $rs1)>;
def : Pat<(int_epi_vmsif_mask EPIVR:$rs1, V0),
        (VMSIF_M_MASK $rs1, vmask_only_true.Value)>;

def : Pat<(int_epi_vmsof EPIVR:$rs1),
        (VMSOF_M $rs1)>;
def : Pat<(int_epi_vmsof_mask EPIVR:$rs1, V0),
        (VMSOF_M_MASK $rs1, vmask_only_true.Value)>;

foreach vtp = AllIntegerVectors.Values in
{
def : Pat<(vtp.Vector (int_epi_vmiota EPIVR:$rs1)),
        (VMIOTA_M $rs1)>;
def : Pat<(vtp.Vector (int_epi_vmiota_mask EPIVR:$rs1, V0)),
        (VMIOTA_M_MASK $rs1, vmask_only_true.Value)>;
}

foreach vtp = AllVectors.Values in
def : Pat<(vtp.Vector (int_epi_vcompress (vtp.Vector EPIVR:$rs2), (nxv1i1 EPIVR:$rs1))),
          (VCOMPRESS_VM $rs2, $rs1)>;

defm "" : pat_intrinsic_binary_int_mm<"int_epi_vmandnot", "VMANDNOT">;
defm "" : pat_intrinsic_binary_int_mm<"int_epi_vmand", "VMAND">;
defm "" : pat_intrinsic_binary_int_mm<"int_epi_vmor", "VMOR">;
defm "" : pat_intrinsic_binary_int_mm<"int_epi_vmxor", "VMXOR">;
defm "" : pat_intrinsic_binary_int_mm<"int_epi_vmornot", "VMORNOT">;
defm "" : pat_intrinsic_binary_int_mm<"int_epi_vmnand", "VMNAND">;
defm "" : pat_intrinsic_binary_int_mm<"int_epi_vmnor", "VMNOR">;
defm "" : pat_intrinsic_binary_int_mm<"int_epi_vmxnor", "VMXNOR">;

defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vdivu", "VDIVU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vdiv", "VDIV">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vremu", "VREMU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vrem", "VREM">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vmulhu", "VMULHU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vmul", "VMUL">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vmulhsu", "VMULHSU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vmulh", "VMULH">;

defm "" : pat_intrinsic_ternary_int_vv_vx<"int_epi_vmadd", "VMADD">;
defm "" : pat_intrinsic_ternary_int_vv_vx<"int_epi_vmsub", "VMSUB">;
defm "" : pat_intrinsic_ternary_int_vv_vx<"int_epi_vmacc", "VMACC">;
defm "" : pat_intrinsic_ternary_int_vv_vx<"int_epi_vmsac", "VMSAC">;

// FIXME: These patterns are wrong
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwaddu", "VWADDU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwadd", "VWADD">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsubu", "VWSUBU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsub", "VWSUB">;

// FIXME: These patterns are wrong
defm "" : pat_intrinsic_binary_int_wv_wx<"int_epi_vwaddu_w", "VWADDU">;
defm "" : pat_intrinsic_binary_int_wv_wx<"int_epi_vwadd_w", "VWADD">;
defm "" : pat_intrinsic_binary_int_wv_wx<"int_epi_vwsubu_w", "VWSUBU">;
defm "" : pat_intrinsic_binary_int_wv_wx<"int_epi_vwsub_w", "VWSUB">;

// FIXME: These patterns are wrong
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmulu", "VWMULU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmulsu", "VWMULSU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmul", "VWMUL">;

// FIXME: These patterns are wrong
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmaccu", "VWMACCU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmacc", "VWMACC">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmsacu", "VWMSACU">;
defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmsac", "VWMSAC">;

defm "" : pat_intrinsic_binary_fp_vv_vf<"int_epi_vfadd", "VFADD">;
defm "" : pat_intrinsic_binary_fp_vs<"int_epi_vfredsum", "VFREDSUM">;
defm "" : pat_intrinsic_binary_fp_vv_vf<"int_epi_vfsub", "VFSUB">;
defm "" : pat_intrinsic_binary_fp_vs<"int_epi_vfredosum", "VFREDOSUM">;
defm "" : pat_intrinsic_binary_fp_vv_vf<"int_epi_vfmin", "VFMIN">;
defm "" : pat_intrinsic_binary_fp_vs<"int_epi_vfredmin", "VFREDMIN">;
defm "" : pat_intrinsic_binary_fp_vv_vf<"int_epi_vfmax", "VFMAX">;
defm "" : pat_intrinsic_binary_fp_vs<"int_epi_vfredmax", "VFREDMAX">;
defm "" : pat_intrinsic_binary_fp_vv_vf<"int_epi_vfsgnj", "VFSGNJ">;
defm "" : pat_intrinsic_binary_fp_vv_vf<"int_epi_vfsgnjn", "VFSGNJN">;
defm "" : pat_intrinsic_binary_fp_vv_vf<"int_epi_vfsgnjx", "VFSGNJX">;

foreach vtp = AllFloatVectors.Values in
def : Pat<(vtp.Scalar (int_epi_vfmv_f_s (vtp.Vector EPIVR:$rs2))),
          (VFMV_F_S $rs2)>;

foreach vtp = AllFloatVectors.Values in
def : Pat<(vtp.Vector (int_epi_vfmv_s_f GPR:$rs2)),
          (VFMV_S_F $rs2)>;

defm "" : pat_intrinsic_binary_relational_fp_vv_vf<"int_epi_vfeq", "VFEQ">;
defm "" : pat_intrinsic_binary_relational_fp_vv_vf<"int_epi_vflte", "VFLTE">;
defm "" : pat_intrinsic_binary_relational_fp_vv_vf<"int_epi_vford", "VFORD">;
defm "" : pat_intrinsic_binary_relational_fp_vv_vf<"int_epi_vflt", "VFLT">;
defm "" : pat_intrinsic_binary_relational_fp_vv_vf<"int_epi_vfne", "VFNE">;
defm "" : pat_intrinsic_binary_relational_fp_vf<"int_epi_vfgt", "VFGT">;
defm "" : pat_intrinsic_binary_relational_fp_vf<"int_epi_vfgte", "VFGTE">;

defm "" : pat_intrinsic_binary_fp_vv_vf<"int_epi_vfdiv", "VFDIV">;
defm "" : pat_intrinsic_binary_fp_vf<"int_epi_vfrdiv", "VFRDIV">;
defm "" : pat_intrinsic_binary_fp_vv_vf<"int_epi_vfmul", "VFMUL">;

// We select VMERGE for a non existing "vfmerge.vv"
defm "" : pat_intrinsic_binary_fp_vv<"int_epi_vfmerge", "VMERGE">;
// vfmerge.vf has its own instruction VFMERGE
defm "" : pat_intrinsic_binary_fp_vf<"int_epi_vfmerge", "VFMERGE">;

defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfmadd", "VFMADD">;
defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfnmadd", "VFNMADD">;
defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfmsub", "VFMSUB">;
defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfnmsub", "VFNMSUB">;
defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfmacc", "VFMACC">;
defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfnmacc", "VFNMACC">;
defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfmsac", "VFMSAC">;
defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfnmsac", "VFNMSAC">;

defm "" : pat_intrinsic_binary_fp_vv_vf<"int_epi_vfwadd", "VFWADD">;
defm "" : pat_intrinsic_binary_fp_vs<"int_epi_vfwredsum", "VFWREDSUM">;
defm "" : pat_intrinsic_binary_fp_vv_vf<"int_epi_vfwsub", "VFWSUB">;
defm "" : pat_intrinsic_binary_fp_vs<"int_epi_vfwredosum", "VFWREDOSUM">;
defm "" : pat_intrinsic_binary_fp_wv_wf<"int_epi_vfwadd_w", "VFWADD">;
defm "" : pat_intrinsic_binary_fp_wv_wf<"int_epi_vfwsub_w", "VFWSUB">;
defm "" : pat_intrinsic_binary_fp_vv<"int_epi_vfdot", "VFDOT">;

defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwmacc", "VFWMACC">;
defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwnmacc", "VFWNMACC">;
defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwmsac", "VFWMSAC">;
defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwnmsac", "VFWNMSAC">;

foreach vtp = AllFloatVectors.Values in
def : Pat<(vtp.Vector (int_epi_vfsqrt (vtp.Vector EPIVR:$rs2))),
          (VFSQRT_V $rs2)>;
foreach vtp = AllFloatVectors.Values in
def : Pat<(vtp.Vector (int_epi_vfsqrt_mask (vtp.Vector EPIVR:$rs2), V0)),
          (VFSQRT_V_MASK $rs2, vmask_only_true.Value)>;

foreach itp = [ nxv1i32, nxv1i64 ] in
foreach vtp = AllFloatVectors.Values in
def : Pat<(itp (int_epi_vfclass (vtp.Vector EPIVR:$rs2))),
          (VFCLASS_V $rs2)>;

foreach itp = [ nxv1i32, nxv1i64 ] in
foreach vtp = AllFloatVectors.Values in
def : Pat<(itp (int_epi_vfclass_mask (vtp.Vector EPIVR:$rs2), V0)),
          (VFCLASS_V_MASK $rs2, vmask_only_true.Value)>;


defm "" : pat_conversions_symmetrical<"VFCVT_XU_F", "vfcvt_xu_f", "VFCVT_F_XU", "vfcvt_f_xu", SameSizePairs.Value>;
defm "" : pat_conversions_symmetrical<"VFCVT_X_F", "vfcvt_x_f", "VFCVT_F_X", "vfcvt_f_x", SameSizePairs.Value>;

defm "" : pat_conversions_symmetrical<"VFWCVT_XU_F", "vfcvt_xu_f", "VFCVT_F_XU", "vfcvt_f_xu", SameSizePairs.Value>;
defm "" : pat_conversions_symmetrical<"VFWCVT_X_F", "vfcvt_x_f", "VFCVT_F_X", "vfcvt_f_x", SameSizePairs.Value>;


defm "" : pat_conversions<"VFWCVT_XU_F", "vfwcvt_xu_f", WidenedSize.Value>;
defm "" : pat_conversions<"VFWCVT_F_XU", "vfwcvt_f_xu", WidenedSize.Value>;
defm "" : pat_conversions<"VFWCVT_X_F", "vfwcvt_x_f", WidenedSize.Value>;
defm "" : pat_conversions<"VFWCVT_F_X", "vfwcvt_f_x", WidenedSize.Value>;
defm "" : pat_conversions<"VFWCVT_F_F", "vfwcvt_f_f", FloatWidenedSize.Value>;

defm "" : pat_conversions<"VFNCVT_XU_F", "vfncvt_xu_f", NarrowedSize.Value>;
defm "" : pat_conversions<"VFNCVT_F_XU", "vfncvt_f_xu", NarrowedSize.Value>;
defm "" : pat_conversions<"VFNCVT_X_F", "vfncvt_x_f", NarrowedSize.Value>;
defm "" : pat_conversions<"VFNCVT_F_X", "vfncvt_f_x", NarrowedSize.Value>;
defm "" : pat_conversions<"VFNCVT_F_F", "vfncvt_f_f", FloatNarrowedSize.Value>;

foreach vtp = AllIntegerVectors.Values in
{
 def : Pat<(vtp.Vector (int_epi_vid)),
            (VID_V)>;
 def : Pat<(vtp.Vector (int_epi_vid_mask V0)),
            (VID_V_MASK vmask_only_true.Value)>;
}

foreach vtp = AllIntegerVectors.Values in {
def : Pat<(vtp.Vector (int_epi_vbroadcast GPR:$rs2)),
          (VMERGE_VX (vtp.Vector (IMPLICIT_DEF)), $rs2)>;
def : Pat<(vtp.Vector (int_epi_vbroadcast simm5:$imm5)),
          (VMERGE_VI (vtp.Vector (IMPLICIT_DEF)), $imm5)>;

def : Pat<(vtp.Vector (int_epi_vbroadcast_mask GPR:$rs2, V0)),
          (VMERGE_VX_MASK (vtp.Vector (IMPLICIT_DEF)), $rs2, vmask_only_true.Value)>;
def : Pat<(vtp.Vector (int_epi_vbroadcast_mask simm5:$imm5, V0)),
          (VMERGE_VI_MASK (vtp.Vector (IMPLICIT_DEF)), $imm5, vmask_only_true.Value)>;
}
foreach vtp = AllFloatVectors.Values in {
def : Pat<(vtp.Vector (int_epi_vbroadcast (vtp.Scalar EPIFPR:$rs2))),
          (VFMERGE_VF (vtp.Vector (IMPLICIT_DEF)), $rs2)>;

def : Pat<(vtp.Vector (int_epi_vbroadcast_mask (vtp.Scalar EPIFPR:$rs2), V0)),
          (VFMERGE_VF_MASK (vtp.Vector (IMPLICIT_DEF)), $rs2, vmask_only_true.Value)>;
}

}

//===----------------------------------------------------------------------===//
// Patterns. Load/store
//===----------------------------------------------------------------------===//

multiclass pat_load<string instruction_width, string intrinsic_type> {

def : Pat<(!cast<Intrinsic>("int_epi_vload_" # intrinsic_type) GPR:$rs1),
          (!cast<Instruction>("VL" # instruction_width # "_V") $rs1)>;
def : Pat<(!cast<Intrinsic>("int_epi_vload_" # intrinsic_type # "_strided")
           GPR:$rs1, GPR:$rs2),
          (!cast<Instruction>("VLS" # instruction_width # "_V") $rs1, $rs2)>;
def : Pat<(!cast<Intrinsic>("int_epi_vload_" # intrinsic_type # "_indexed")
           GPR:$rs1, EPIVR:$rs2),
          (!cast<Instruction>("VLX" # instruction_width # "_V") $rs1, $rs2)>;
def : Pat<(!cast<Intrinsic>("int_epi_vload_" # intrinsic_type # "_mask")
           GPR:$rs1, V0),
           (!cast<Instruction>("VL" # instruction_width # "_V_MASK") $rs1, vmask_only_true.Value)>;
def : Pat<(!cast<Intrinsic>("int_epi_vload_" # intrinsic_type # "_strided_mask")
           GPR:$rs1, GPR:$rs2, V0),
          (!cast<Instruction>("VLS" # instruction_width # "_V_MASK")
           $rs1, $rs2, vmask_only_true.Value)>;
def : Pat<(!cast<Intrinsic>("int_epi_vload_" # intrinsic_type # "_indexed_mask")
           GPR:$rs1, EPIVR:$rs2, V0),
          (!cast<Instruction>("VLX" # instruction_width # "_V_MASK")
           $rs1, $rs2, vmask_only_true.Value)>;

}


multiclass pat_store<string instruction_width, string intrinsic_type> {

def : Pat<(!cast<Intrinsic>("int_epi_vstore_" # intrinsic_type)
           EPIVR:$rs3, GPR:$rs1),
          (!cast<Instruction>("VS" # instruction_width # "_V") $rs3, $rs1)>;
def : Pat<(!cast<Intrinsic>("int_epi_vstore_" # intrinsic_type # "_strided")
           EPIVR:$rs3, GPR:$rs1, GPR:$rs2),
          (!cast<Instruction>("VSS" # instruction_width # "_V") $rs3, $rs1, $rs2)>;
def : Pat<(!cast<Intrinsic>("int_epi_vstore_" # intrinsic_type # "_indexed")
           EPIVR:$rs3, GPR:$rs1, EPIVR:$rs2),
          (!cast<Instruction>("VSX" # instruction_width # "_V") $rs3, $rs1, $rs2)>;
def : Pat<(!cast<Intrinsic>("int_epi_vstore_" # intrinsic_type # "_mask")
           EPIVR:$rs3, GPR:$rs1, V0),
           (!cast<Instruction>("VS" # instruction_width # "_V_MASK") $rs3, $rs1, vmask_only_true.Value)>;
def : Pat<(!cast<Intrinsic>("int_epi_vstore_" # intrinsic_type # "_strided_mask")
           EPIVR:$rs3, GPR:$rs1, GPR:$rs2, V0),
          (!cast<Instruction>("VSS" # instruction_width # "_V_MASK")
           $rs3, $rs1, $rs2, vmask_only_true.Value)>;
def : Pat<(!cast<Intrinsic>("int_epi_vstore_" # intrinsic_type # "_indexed_mask")
           EPIVR:$rs3, GPR:$rs1, EPIVR:$rs2, V0),
          (!cast<Instruction>("VSX" # instruction_width # "_V_MASK")
           $rs3, $rs1, $rs2, vmask_only_true.Value)>;

}

let Predicates = [HasExtEPI] in {

defm "" : pat_load<"B", "i8">;
defm "" : pat_load<"H", "i16">;
defm "" : pat_load<"W", "i32">;
defm "" : pat_load<"E", "i64">;

defm "" : pat_load<"W", "f32">;
defm "" : pat_load<"E", "f64">;

defm "" : pat_store<"B", "i8">;
defm "" : pat_store<"H", "i16">;
defm "" : pat_store<"W", "i32">;
defm "" : pat_store<"E", "i64">;

defm "" : pat_store<"W", "f32">;
defm "" : pat_store<"E", "f64">;

}
