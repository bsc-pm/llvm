//=- RISCVInstrInfoV.td - Zeou-Extension RISCV instructions -*- tblgen-*----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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

let Predicates = [HasStdExtV] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 0,
    Uses = [VL] in
def PseudoReadVL : Pseudo<(outs GPR:$rd),
                          (ins), [], "rdvl", "$rd">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1,
     usesCustomInserter = 1 in
def PseudoVSCALE : Pseudo<(outs GPR:$rd), (ins), [], "">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 0,
    Uses = [VTYPE] in
def PseudoReadVTYPE : Pseudo<(outs GPR:$rd),
                          (ins), [], "rdvtype", "$rd">;

//===----------------------------------------------------------------------===//
// Pseudo instructions we need for SPILL and RELOAD
//===----------------------------------------------------------------------===//

let hasSideEffects = 1, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1,
    Uses = [VL, VTYPE] in
def PseudoVSPILL : Pseudo<(outs), (ins VR:$rs1, GPR:$rs2), []>;

let hasSideEffects = 1, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1,
    Uses = [VL, VTYPE] in
def PseudoVRELOAD : Pseudo<(outs VR:$rs1), (ins GPR:$rs2), []>;

}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

class VectorTypeInfo<ValueType Vec, ValueType Mas, int Sew, VReg Reg>
{
  ValueType Vector = Vec;
  ValueType Mask = Mas;
  int SEW = Sew;
  VReg RegClass = Reg;
}

class FloatVectorTypeInfo<ValueType Vec, ValueType Mas, int Sew, VReg Reg,
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
//    def Vtype1xi8  : VectorTypeInfo<nxv1i8,  nxv1i1, 8,  VR>; // FIXME illegal type
//    def Vtype2xi8  : VectorTypeInfo<nxv2i8,  nxv2i1, 8,  VR>; // FIXME illegal type
//    def Vtype4xi8  : VectorTypeInfo<nxv4i8,  nxv4i1, 8,  VR>; // FIXME illegal type
      def Vtype8xi8  : VectorTypeInfo<nxv8i8,  nxv8i1, 8,  VR>;

//    def Vtype1xi16 : VectorTypeInfo<nxv1i16, nxv1i1, 16, VR>; // FIXME illegal type
//    def Vtype2xi16 : VectorTypeInfo<nxv2i16, nxv2i1, 16, VR>; // FIXME illegal type
      def Vtype4xi16 : VectorTypeInfo<nxv4i16, nxv4i1, 16, VR>;

//    def Vtype1xi32 : VectorTypeInfo<nxv1i32, nxv1i1, 32, VR>; // FIXME illegal type
      def Vtype2xi32 : VectorTypeInfo<nxv2i32, nxv2i1, 32, VR>;

      def Vtype1xi64 : VectorTypeInfo<nxv1i64, nxv1i1, 64, VR>;
    }

    defset list<VectorTypeInfo> GroupIntegerVectors = {
      def Vtype16xi8  : VectorTypeInfo<nxv16i8,  nxv16i1, 8,  VR2>;
      def Vtype32xi8  : VectorTypeInfo<nxv32i8,  nxv32i1, 8,  VR4>;
//    def Vtype64xi8  : VectorTypeInfo<nxv64i8,  nxv64i1, 8,  VR8>; // FIXME undefined type

      def Vtype8xi16  : VectorTypeInfo<nxv8i16,  nxv8i1,  16, VR2>;
      def Vtype16xi16 : VectorTypeInfo<nxv16i16, nxv16i1, 16, VR4>;
      def Vtype32xi16 : VectorTypeInfo<nxv32i16, nxv32i1, 16, VR8>;

      def Vtype4xi32  : VectorTypeInfo<nxv4i32,  nxv4i1,  32, VR2>;
      def Vtype8xi32  : VectorTypeInfo<nxv8i32,  nxv8i1,  32, VR4>;
      def Vtype16xi32 : VectorTypeInfo<nxv16i32, nxv16i1, 32, VR8>;

      def Vtype2xi64  : VectorTypeInfo<nxv2i64,  nxv2i1,  64, VR2>;
      def Vtype4xi64  : VectorTypeInfo<nxv4i64,  nxv4i1,  64, VR4>;
      def Vtype8xi64  : VectorTypeInfo<nxv8i64,  nxv8i1,  64, VR8>;
    }
  }

  defset list<FloatVectorTypeInfo> AllFloatVectors = {
    defset list<FloatVectorTypeInfo> NoGroupFloatVectors = {
//    def Vtype1xf32 : FloatVectorTypeInfo<nxv1f32, nxv1i1, 32, VR, f32, FPR32>; // FIXME illegal type
      def Vtype2xf32 : FloatVectorTypeInfo<nxv2f32, nxv2i1, 32, VR, f32, FPR32>;

      def Vtype1xf64 : FloatVectorTypeInfo<nxv1f64, nxv1i1, 64, VR, f64, FPR64>;
    }

    def Vtype4xf32  : FloatVectorTypeInfo<nxv4f32,  nxv4i1,  32, VR2, f32, FPR32>;
    def Vtype8xf32  : FloatVectorTypeInfo<nxv8f32,  nxv8i1,  32, VR4, f32, FPR32>;
    def Vtype16xf32 : FloatVectorTypeInfo<nxv16f32, nxv16i1, 32, VR8, f32, FPR32>;

    def Vtype2xf64  : FloatVectorTypeInfo<nxv2f64,  nxv2i1,  64, VR2, f64, FPR64>;
    def Vtype4xf64  : FloatVectorTypeInfo<nxv4f64,  nxv4i1,  64, VR4, f64, FPR64>;
    def Vtype8xf64  : FloatVectorTypeInfo<nxv8f64,  nxv8i1,  64, VR8, f64, FPR64>;
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

class VRegToWide<VReg Reg, VReg WideReg> {
  VReg RegClass = Reg;
  VReg WideRegClass = WideReg;
}

defset list<VRegToWide> AllVRegToWidePairs = {
  def : VRegToWide<VR, VR2>;
  def : VRegToWide<VR2, VR4>;
  def : VRegToWide<VR4, VR8>;
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


multiclass pseudo_nullary<VReg result_reg_class,
                          int vlmul,
                          string constraints = ""> {
  let Constraints = Join<[constraints, "$rd = $merge"], ",">.ret,
      Uses = [VL, VTYPE], VLIndex = 3, SEWIndex = 4, MergeOpIndex = 1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins result_reg_class:$merge,
                                 VRMaskRegOp:$vm,
                                 GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

multiclass pseudo_nullary_v {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _V : pseudo_nullary<evr, evr.VLMul>;
    }
  }
}

multiclass pseudo_unary<VReg result_reg_class,
                        VReg op_reg_class,
                        int vlmul,
                        string constraints = ""> {
  let Constraints = Join<[constraints, "$rd = $merge"], ",">.ret,
      Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = 1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins result_reg_class:$merge,
                                 op_reg_class:$rs2, VRMaskRegOp:$vm,
                                 GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

// Special case for masking that does not have a merge operand.
multiclass pseudo_unary_nomerge<VReg result_reg_class,
                                VReg op_reg_class,
                                int vlmul,
                                string constraints = ""> {
  let Constraints = constraints,
      Uses = [VL, VTYPE], VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins op_reg_class:$rs2, VRMaskRegOp:$vm,
                                 GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

// Special case for masking that does not have a merge operand nor a mask.
multiclass pseudo_unary_nomask<VReg result_reg_class,
                               VReg op_reg_class,
                               int vlmul,
                               string constraints = ""> {
  let Constraints = constraints,
      Uses = [VL, VTYPE], VLIndex = 2, SEWIndex = 3, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins op_reg_class:$rs2,
                                 GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

multiclass pseudo_unary_v_v {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
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
  foreach evr = AllVRegs in
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
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _F : pseudo_unary_nomask<evr, FPR32, evr.VLMul>;
    }
  }
}

multiclass pseudo_unary_w_v {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegToWidePairs in
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
  foreach evr = AllVRegToWidePairs in
  {
    let VLMul = evr.RegClass.VLMul in
    {
      defm _W : pseudo_unary<evr.RegClass, evr.WideRegClass,
                             evr.RegClass.VLMul, "@earlyclobber $rd">;
    }
  }
}

multiclass pseudo_binary<VReg result_reg_class,
                         VReg op1_reg_class,
                         RegisterClass op2_reg_class, // FIXME dagoperand
                         int vlmul, string constraints = ""> {
  let Constraints = Join<[constraints, "$rd = $merge"], ",">.ret,
      Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins result_reg_class:$merge,
                                 op1_reg_class:$rs2, op2_reg_class:$rs1,
                                 VRMaskRegOp:$vm, GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

// Special case for masking that does not have a merge operand nor a mask.
multiclass pseudo_binary_nomask<VReg result_reg_class,
                                VReg op1_reg_class,
                                RegisterClass op2_reg_class, // FIXME dagoperand
                                int vlmul, string constraints = ""> {
  let Constraints = constraints,
      Uses = [VL, VTYPE], VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins op1_reg_class:$rs2, op2_reg_class:$rs1,
                                 GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

multiclass pseudo_binary_mask_in<VReg result_reg_class,
                                 VReg op1_reg_class,
                                 RegisterClass op2_reg_class,
                                 int vlmul> {
  let Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins op1_reg_class:$rs2, op2_reg_class:$rs1,
                                 VRV0:$maskop, GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

multiclass pseudo_binary_v_vv {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<evr, evr, evr, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_v_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VX : pseudo_binary<evr, evr, GPR, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_v_vs {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VS : pseudo_binary<evr, evr, evr, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_v_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
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
  foreach evr = AllVRegs in
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
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VX : pseudo_binary<evr, evr, GPR, evr.VLMul>;
      defm _VI : pseudo_binary<evr, evr, uimm5, evr.VLMul>; // FIXME how can this work? uimm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_v_vv_vx_vi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<evr, evr, evr, evr.VLMul>;
      defm _VX : pseudo_binary<evr, evr, GPR, evr.VLMul>;
      defm _VI : pseudo_binary<evr, evr, simm5, evr.VLMul>; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_v_wv_wx_wi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegToWidePairs in
  {
    let VLMul = evr.RegClass.VLMul in
    {
      defm _WV : pseudo_binary<evr.RegClass, evr.WideRegClass, evr.RegClass,
                               evr.RegClass.VLMul, "@earlyclobber $rd">;
      defm _WX : pseudo_binary<evr.RegClass, evr.WideRegClass, GPR,
                               evr.RegClass.VLMul, "@earlyclobber $rd">;
      defm _WI : pseudo_binary<evr.RegClass, evr.WideRegClass, simm5,
                               evr.RegClass.VLMul, "@earlyclobber $rd">; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_v_vvm_vxm_vim {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
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
  foreach evr = AllVRegToWidePairs in
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
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<VR, evr, evr, evr.VLMul>;
      defm _VX : pseudo_binary<VR, evr, GPR, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_m_vx_vi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VX : pseudo_binary<VR, evr, GPR, evr.VLMul>;
      defm _VI : pseudo_binary<VR, evr, simm5, evr.VLMul>; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_m_vx_vu {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VX : pseudo_binary<VR, evr, GPR, evr.VLMul>;
      defm _VI : pseudo_binary<VR, evr, uimm5, evr.VLMul>; // FIXME how can this work? uimm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_m_vv_vx_vi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<VR, evr, evr, evr.VLMul>;
      defm _VX : pseudo_binary<VR, evr, GPR, evr.VLMul>;
      defm _VI : pseudo_binary<VR, evr, simm5, evr.VLMul>; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_m_vv_vx_vu {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<VR, evr, evr, evr.VLMul>;
      defm _VX : pseudo_binary<VR, evr, GPR, evr.VLMul>;
      defm _VI : pseudo_binary<VR, evr, uimm5, evr.VLMul>; // FIXME how can this work? simm5 is not a RegisterClass but an Operand
    }
  }
}

multiclass pseudo_binary_v_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VF : pseudo_binary<evr, evr, FPR32, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_v_vfm {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VFM : pseudo_binary_mask_in<evr, evr, FPR32, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_v_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<evr, evr, evr, evr.VLMul>;
      defm _VF : pseudo_binary<evr, evr, FPR32, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_w_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegToWidePairs in
  {
    let VLMul = evr.RegClass.VLMul in
    {
      defm _VV : pseudo_binary<evr.WideRegClass, evr.RegClass, evr.RegClass,
                               evr.RegClass.VLMul, "@earlyclobber $rd">;
      defm _VF : pseudo_binary<evr.WideRegClass, evr.RegClass, FPR32,
                               evr.RegClass.VLMul, "@earlyclobber $rd">;
    }
  }
}

multiclass pseudo_binary_m_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VF : pseudo_binary<VR, evr, FPR32, evr.VLMul>;
    }
  }
}

multiclass pseudo_binary_m_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<VR, evr, evr, evr.VLMul>;
      defm _VF : pseudo_binary<VR, evr, FPR32, evr.VLMul>;
    }
  }
}

multiclass pseudo_ternary<VReg result_reg_class,
                          VReg op1_reg_class,
                          RegisterClass op2_reg_class,
                          int vlmul> { // FIXME dagoperand?
  let Constraints = "$rd = $rs3",
      Uses = [VL, VTYPE],
      VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                                 (ins result_reg_class:$rs3, op1_reg_class:$rs1,
                                      op2_reg_class:$rs2, VRMaskRegOp:$vm,
                                      GPR:$vl, ixlenimm:$sew),
                                 []>,
                          EPIPseudo;
}

multiclass pseudo_ternary_v_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
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
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_ternary<evr, evr, evr, evr.VLMul>;
      defm _VF : pseudo_ternary<evr, FPR32, evr, evr.VLMul>;
    }
  }
}

multiclass pseudo_unary_x_v {
  // A different pseudo is defined for each VLMul value, but (ungrouped) VR
  // register operands are used in all of them.
  foreach Vlmul = [1, 2, 4, 8] in
  {
    let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1,
        VLMul = Vlmul in
    {
      defm _M : pseudo_unary_nomerge<GPR, VR, Vlmul>;
    }
  }
}

multiclass pseudo_binary_m_mm {
  // A different pseudo is defined for each VLMul value, but (ungrouped) VR
  // register operands are used in all of them.
  foreach Vlmul = [1, 2, 4, 8] in
  {
    let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1,
        VLMul = Vlmul in
    {
      defm _MM : pseudo_binary_nomask<VR, VR, VR, Vlmul>;
    }
  }
}

let Predicates = [HasStdExtV] in {

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
defm PseudoVAADD       : pseudo_binary_v_vv_vx;
// FIXME missing PseudoVAADDU
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
    Uses = [VL, VTYPE], VLMul = VR.VLMul, MergeOpIndex = -1 in
{
  let VLIndex = 2, SEWIndex = 3, BaseInstr = VMV_X_S in
    def PseudoVMV_X_S : Pseudo<(outs GPR:$rd),
                               (ins VR:$rs2, GPR:$vl, ixlenimm:$sew),
                               []>,
                        EPIPseudo;
  let VLIndex = 2, SEWIndex = 3, BaseInstr = VMV_S_X in
    def PseudoVMV_S_X : Pseudo<(outs VR:$rd),
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
defm PseudoVNMSUB      : pseudo_ternary_v_vv_vx;
defm PseudoVMACC       : pseudo_ternary_v_vv_vx;
defm PseudoVNMSAC      : pseudo_ternary_v_vv_vx;

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
    Uses = [VL, VTYPE], VLMul = VR.VLMul, MergeOpIndex = -1 in
{
  let VLIndex = 2, SEWIndex = 3, BaseInstr = VFMV_F_S in
    def PseudoVFMV_F_S : Pseudo<(outs FPR32:$rd),
                                (ins VR:$rs2, GPR:$vl, ixlenimm:$sew),
                                []>,
                         EPIPseudo;
  let VLIndex = 2, SEWIndex = 3, BaseInstr = VFMV_S_F in
    def PseudoVFMV_S_F : Pseudo<(outs VR:$rd),
                                (ins FPR32:$rs2, GPR:$vl, ixlenimm:$sew),
                                []>,
                         EPIPseudo;
}

defm PseudoVMFEQ       : pseudo_binary_m_vv_vf;
defm PseudoVMFLE       : pseudo_binary_m_vv_vf;
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

defm PseudoVPOPC       : pseudo_unary_x_v;
defm PseudoVFIRST      : pseudo_unary_x_v;

defm PseudoVMANDNOT    : pseudo_binary_m_mm;
defm PseudoVMAND       : pseudo_binary_m_mm;
defm PseudoVMOR        : pseudo_binary_m_mm;
defm PseudoVMXOR       : pseudo_binary_m_mm;
defm PseudoVMORNOT     : pseudo_binary_m_mm;
defm PseudoVMNAND      : pseudo_binary_m_mm;
defm PseudoVMNOR       : pseudo_binary_m_mm;
defm PseudoVMXNOR      : pseudo_binary_m_mm;

// Load/store pseudo instructions

foreach evr = AllVRegs in
foreach vlmul = [evr.VLMul] in
{
  let mayLoad = 1, mayStore = 0, hasSideEffects = 0,
      usesCustomInserter = 1,
      VLMul = vlmul in
  {
    let Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = 1,
        BaseInstr = VLE_V in
      def PseudoVLE_V_M#vlmul
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, VRMaskRegOp:$mask, GPR:$vl,
                    ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
        BaseInstr = VLSE_V in
      def PseudoVLSE_V_M#vlmul
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, GPR:$rs2, VRMaskRegOp:$mask,
                    GPR:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
        BaseInstr = VLXE_V in
      def PseudoVLXE_V_M#vlmul
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, evr:$rs2, VRMaskRegOp:$mask,
                    GPR:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;
  }

  let mayLoad = 0, mayStore = 1, hasSideEffects = 0,
      usesCustomInserter = 1,
      VLMul = vlmul in
  {
    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE],
        VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
        BaseInstr = VSE_V in
      def PseudoVSE_V_M#vlmul
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, VRMaskRegOp:$mask, GPR:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo;

    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE],
        VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
        BaseInstr = VSSE_V in
      def PseudoVSSE_V_M#vlmul
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, GPR:$rs2, VRMaskRegOp:$mask, GPR:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo;

    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE],
        VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
        BaseInstr = VSXE_V in
      def PseudoVSXE_V_M#vlmul
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, evr:$rs2, VRMaskRegOp:$mask, GPR:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo;
  }
}

}

//===----------------------------------------------------------------------===//
// Patterns. Essential
//===----------------------------------------------------------------------===//

// Note: PatLeaf can't be used here as it is considered an input pattern.
def VLMax : OutPatFrag<(ops), (i64 X0)>;

// Floating point instructions with a scalar operand expect such operand to be
// in a register of class FPR32. When dealing with the f64 variant of such
// instructions we need to extract the FPR32 subregister from the FPR64 base
// register to match the instruction operand
class ToFPR32<ValueType type, DAGOperand operand, string name> {
  dag ret = !if(!eq(!cast<string>(operand),
                    !cast<string>(FPR64)),
                (EXTRACT_SUBREG !dag(type, [FPR64], [name]), sub_32),
                !dag(type, [operand], [name]));
}

multiclass pat_load_store<LLVMType type,
                          LLVMType mask_type,
                          string load_instr_name,
                          string store_instr_name,
                          int sew,
                          int vlmul,
                          VReg reg_class>
{
  // Load
  def : Pat<(type (load GPR:$rs1)),
            (!cast<Instruction>(load_instr_name#"_M"#vlmul)
             (type zero_reg),
             GPR:$rs1,
             (mask_type zero_reg),
             VLMax, sew)>;
  def : Pat<(type (load AddrFI:$rs1)),
            (!cast<Instruction>(load_instr_name#"_M"#vlmul)
             (type zero_reg),
             AddrFI:$rs1,
             (mask_type zero_reg),
             VLMax, sew)>;

  // Store
  def : Pat<(store type:$rs2, GPR:$rs1),
            (!cast<Instruction>(store_instr_name#"_M"#vlmul)
             reg_class:$rs2, GPR:$rs1,
             (mask_type zero_reg),
             VLMax, sew)>;
  def : Pat<(store type:$rs2, AddrFI:$rs1),
            (!cast<Instruction>(store_instr_name#"_M"#vlmul)
             reg_class:$rs2, AddrFI:$rs1,
             (mask_type zero_reg),
             VLMax, sew)>;
}

// Extra patterns for truncating loads and extending stores of mask types
multiclass pat_load_store_mask<LLVMType type,
                               LLVMType integer_type,
                               string load_instr_name,
                               string store_instr_name,
                               int sew,
                               int vlmul,
                               VReg reg_class>
{
  // Load
  def : Pat<(type (trunc (integer_type (load GPR:$rs1)))),
            (!cast<Instruction>(load_instr_name#"_M"#vlmul)
             (type zero_reg),
             GPR:$rs1,
             (type zero_reg),
             VLMax, sew)>;
  def : Pat<(type (trunc (integer_type (load AddrFI:$rs1)))),
            (!cast<Instruction>(load_instr_name#"_M"#vlmul)
             (type zero_reg),
             AddrFI:$rs1,
             (type zero_reg),
             VLMax, sew)>;

  // Store
  def : Pat<(store (integer_type (zext type:$rs2)), GPR:$rs1),
            (!cast<Instruction>(store_instr_name#"_M"#vlmul)
             reg_class:$rs2, GPR:$rs1,
             (type zero_reg),
             VLMax, sew)>;
  def : Pat<(store (integer_type (zext type:$rs2)), AddrFI:$rs1),
            (!cast<Instruction>(store_instr_name#"_M"#vlmul)
             reg_class:$rs2, AddrFI:$rs1,
             (type zero_reg),
             VLMax, sew)>;
}

let Predicates = [HasStdExtV] in {
  foreach vti = AllVectors in
  {
    defm : pat_load_store<vti.Vector, vti.Mask, "PseudoVLE_V", "PseudoVSE_V",
                          vti.SEW, vti.RegClass.VLMul, vti.RegClass>;
  }

  foreach mti = AllMasks in
  {
    // Mask values are stored as vectors of bytes
    defm : pat_load_store_mask<mti.Mask, mti.IntegerVector,
                               "PseudoVLE_V", "PseudoVSE_V",
                               /* sew */ 8, /* vlmul */ 1, VR>;
  }

  foreach vti = NoGroupIntegerVectors in {
     def : Pat<(vti.Vector (zext (vti.Mask VR:$rs1))),
               (PseudoVMAND_MM_M1 VR:$rs1, VR:$rs1, VLMax, vti.SEW)>;
     // We don't have to do anything here because mask vectors only use the
     // LSB of each element.
     def : Pat<(vti.Mask (trunc (vti.Vector VR:$rs1))),
               (COPY_TO_REGCLASS VR:$rs1, VR)>;
  }

  foreach vti = GroupIntegerVectors in {
     def : Pat<(vti.Mask (trunc (vti.Vector vti.RegClass:$rs1))),
               (!cast<Instruction>("PseudoVMSNE_VV_M"#vti.RegClass.VLMul)
                (vti.Mask zero_reg),
                (!cast<Instruction>("PseudoVAND_VI_M"#vti.RegClass.VLMul)
                 (vti.Vector zero_reg),
                 (vti.Vector vti.RegClass:$rs1), 1,
                 (vti.Mask zero_reg),
                 VLMax, vti.SEW),
                (!cast<Instruction>("PseudoVAND_VI_M"#vti.RegClass.VLMul)
                 (vti.Vector zero_reg),
                 (vti.Vector (IMPLICIT_DEF)), 0,
                 (vti.Mask zero_reg),
                 VLMax, vti.SEW),
                (vti.Mask zero_reg),
                VLMax, vti.SEW)>;
     def : Pat<(vti.Vector (zext (vti.Mask V0))),
               (!cast<Instruction>("PseudoVMERGE_VIM_M"#vti.RegClass.VLMul)
                (!cast<Instruction>("PseudoVAND_VI_M"#vti.RegClass.VLMul)
                 (vti.Vector zero_reg),
                 (vti.Vector (IMPLICIT_DEF)), 0,
                 (vti.Mask zero_reg),
                 VLMax, vti.SEW),
                1,
                (vti.Mask V0),
                VLMax, vti.SEW)>;
  }

  // FIXME Deal with {B,H,W} load/store (load + sext)
}

//===----------------------------------------------------------------------===//
// Patterns. Common
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

def pow2uimm5 : ImmLeaf<XLenVT, [{
    return isPowerOf2_64(Imm) && isUInt<5>(Log2_64(Imm));
}]>;
def Log2 : SDNodeXForm<imm,
[{
    return CurDAG->getTargetConstant(Log2_64(N->getZExtValue()), SDLoc(N),
        MVT::i64);
}]>;

let Predicates = [HasStdExtV] in {

def : Pat<(int_vscale), (PseudoVSCALE)>;

// FIXME: These patterns can be improved.
def : Pat<(vscale 1), (PseudoVSCALE)>;
def : Pat<(vscale pow2uimm5:$mul),
          (SLLI (PseudoVSCALE), (Log2 imm:$mul))>;

// FIXME: uimm5 here is not very precise. It should be uimm3 and uimm2 but we
// will probably want to wait for intrinsics with constant values constraints.
// FIXME This pattern is wrong
//def : Pat<(int_epi_vsetvl GPR:$vl, uimm5:$sew, uimm5:$vmul),
//          (VSETVLI (NoX0 GPR:$vl), uimm5:$sew, uimm5:$vmul)>;

// FIXME: uimm5 here is not very precise. It should be uimm3 and uimm2 but we
// will probably want to wait for intrinsics with constant values constraints.
// FIXME This pattern is wrong
//def : Pat<(int_epi_vsetvlmax uimm5:$sew, uimm5:$vmul),
//          (VSETVLI X0, uimm5:$sew, uimm5:$vmul)>;

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

multiclass pat_intrinsic_binary<string intrinsic_name,
                                string instruction_name,
                                string kind,
                                ValueType result_type,
                                ValueType op1_type,
                                ValueType op2_type,
                                ValueType mask_type,
                                int sew,
                                int vlmul,
                                VReg result_reg_class,
                                VReg op1_reg_class,
                                RegisterClass op2_reg_class, // FIXME dagoperand?
                                bit swap = 0>
{
  foreach instruction = [!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)] in
  {
    def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                            (op1_type op1_reg_class:$rs1),
                            (op2_type op2_reg_class:$rs2),
                            (i64 GPR:$vl))),
              swap_helper<
                (instruction (result_type zero_reg)),
                (instruction
                 (op1_type op1_reg_class:$rs1)),
                (instruction ToFPR32<op2_type, op2_reg_class, "rs2">.ret),
                (instruction
                 (mask_type zero_reg),
                 (NoX0 GPR:$vl),
                 sew),
                swap>.Value>;

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
                (instruction ToFPR32<op2_type, op2_reg_class, "rs2">.ret),
                (instruction
                 (mask_type V0),
                 (NoX0 GPR:$vl),
                 sew),
                swap>.Value>;
  }
}

multiclass pat_intrinsic_binary_nomask<string intrinsic_name,
                                       string instruction_name,
                                       string kind,
                                       ValueType result_type,
                                       ValueType op1_type,
                                       ValueType op2_type,
                                       ValueType mask_type,
                                       int sew,
                                       int vlmul,
                                       VReg op1_reg_class,
                                       RegisterClass op2_reg_class, // FIXME dagoperand?
                                       bit swap = 0>
{
  foreach instruction = [!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)] in
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2),
                          (i64 GPR:$vl))),
            swap_helper<
              (instruction), // empty prefix
              (instruction
               (op1_type op1_reg_class:$rs1)),
              (instruction ToFPR32<op2_type, op2_reg_class, "rs2">.ret),
              (instruction
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
                                        VReg result_reg_class,
                                        VReg op1_reg_class,
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
              (instruction ToFPR32<op2_type, op2_reg_class, "rs2">.ret),
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
                                vti.SEW, vti.RegClass.VLMul, VR,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_relational_int_m_vv_vx_vu<string intrinsic_name,
                                                          string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, VR,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_relational_int_m_vv_vx<string intrinsic_name,
                                                       string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, VR,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, GPR>;
  }
}

multiclass pat_intrinsic_binary_relational_int_m_vx_vi<string intrinsic_name,
                                                       string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_relational_int_m_vx_vu<string intrinsic_name,
                                                       string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, uimm5>;
  }
}

multiclass pat_intrinsic_binary_relational_int_v_vv_swapped<string intrinsic_name,
                                                            string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, VR,
                                vti.RegClass, vti.RegClass, /* swapped */ 1>;
  }
}

//multiclass pat_intrinsic_binary_with_mask_input_vv_nomask<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType result_type>
//{
//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type VR:$rs1), (op_type VR:$rs2), (op_type V0))),
//            (!cast<Instruction>(instruction_name # "_VV")
//               VR:$rs1, VR:$rs2)>;
//}

//multiclass pat_intrinsic_binary_with_mask_input_vx_nomask<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType result_type>
//{
//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type VR:$rs1), GPR:$rs2, (op_type V0))),
//            (!cast<Instruction>(instruction_name # "_VX")
//               VR:$rs1, GPR:$rs2)>;
//}

//multiclass pat_intrinsic_binary_with_mask_input_vi_nomask<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType result_type>
//{
//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type VR:$rs1), simm5:$imm5, (op_type V0))),
//            (!cast<Instruction>(instruction_name # "_VI")
//               VR:$rs1, simm5:$imm5)>;
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
                                fvti.SEW, fvti.RegClass.VLMul, VR,
                                fvti.RegClass, fvti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Mask, fvti.Vector, fvti.Scalar, fvti.Mask,
                                fvti.SEW, fvti.RegClass.VLMul, VR,
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
                                fvti.SEW, fvti.RegClass.VLMul, VR,
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
                                fvti.SEW, fvti.RegClass.VLMul, VR,
                                fvti.RegClass, fvti.RegClass, /* swapped */ 1>;
  }
}

multiclass pat_intrinsic_binary_m_mm<string intrinsic_name,
                                     string instruction_name>
{
  foreach mti = AllMasks in
  {
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "MM",
                                       mti.Mask, mti.Mask, mti.Mask, mti.Mask,
                                       mti.SEW, mti.VLMul, VR, VR>;
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
                                 VReg result_reg_class,
                                 VReg op1_reg_class,
                                 VReg op2_reg_class> // FIXME Registerclass, dagoperand?
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (result_type result_reg_class:$rs3),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2),
                          (i64 GPR:$vl))),
            (!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)
             result_reg_class:$rs3,
             ToFPR32<op1_type, op1_reg_class, "rs1">.ret,
             op2_reg_class:$rs2,
             (mask_type zero_reg),
             (NoX0 GPR:$vl),
             sew)>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                          (result_type result_reg_class:$rs3),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2),
                          (mask_type V0),
                          (i64 GPR:$vl))),
            (!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)
             result_reg_class:$rs3,
             ToFPR32<op1_type, op1_reg_class, "rs1">.ret,
             op2_reg_class:$rs2,
             (mask_type V0),
             (NoX0 GPR:$vl),
             sew)>;
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
//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type VR:$rs1), (op_type VR:$rs2))),
//            (!cast<Instruction>(instruction_name # "_WV")
//               VR:$rs1, VR:$rs2)>;
//}

//multiclass pat_intrinsic_binary_wv<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType result_type>
//{
//  defm "" : pat_intrinsic_binary_wv_nomask<intrinsic_name, instruction_name,
//                                           op_type, result_type>;

//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
//                         (op_type VR:$rs1),
//                         (op_type VR:$rs2), V0)),
//            (!cast<Instruction>(instruction_name # "_WV_MASK")
//               VR:$rs1, VR:$rs2, vmask_only_true.Value)>;
//}

//multiclass pat_intrinsic_binary_wf_nomask<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType op_scalar,
//                                   ValueType result_type>
//{
//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type VR:$rs1), (op_scalar EPIFPR:$rs2))),
//            (!cast<Instruction>(instruction_name # "_WF")
//               VR:$rs1, EPIFPR:$rs2)>;
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
//                         (op_type VR:$rs1),
//                         (op_scalar EPIFPR:$rs2), V0)),
//            (!cast<Instruction>(instruction_name # "_WF_MASK")
//               VR:$rs1, EPIFPR:$rs2, vmask_only_true.Value)>;
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
//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name) (op_type VR:$rs1), GPR:$rs2)),
//            (!cast<Instruction>(instruction_name # "_WX")
//               VR:$rs1, GPR:$rs2)>;
//}

//multiclass pat_intrinsic_binary_wx<string intrinsic_name,
//                                   string instruction_name,
//                                   ValueType op_type,
//                                   ValueType result_type>
//{
//  defm "" : pat_intrinsic_binary_wx_nomask<intrinsic_name, instruction_name,
//                                           op_type, result_type>;

//  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
//                         (op_type VR:$rs1),
//                         GPR:$rs2, V0)),
//            (!cast<Instruction>(instruction_name # "_WX_MASK")
//               VR:$rs1, GPR:$rs2, vmask_only_true.Value)>;
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
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_any_and_int_v_vx_vi<string intrinsic_name,
                                                    string instruction_name>
{
  foreach vti = AllVectors in
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
//def : Pat<(vtp.First (!cast<Intrinsic>("int_epi_" # intrinsic) (vtp.Second VR:$rs1))),
//          (!cast<Instruction>(instruction # "_V") VR:$rs1)>;
//def : Pat<(vtp.First (!cast<Intrinsic>("int_epi_" # intrinsic # "_mask") (vtp.Second VR:$rs1), V0)),
//          (!cast<Instruction>(instruction # "_V_MASK") VR:$rs1, vmask_only_true.Value)>;

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
                               VReg result_reg_class,
                               VReg op1_reg_class> // FIXME Registerclass, dagoperand?
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (i64 GPR:$vl))),
            (!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)
             (result_type zero_reg),
             op1_reg_class:$rs1,
             (mask_type zero_reg),
             (NoX0 GPR:$vl), sew)>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                          (result_type result_reg_class:$merge),
                          (op1_type op1_reg_class:$rs1),
                          (mask_type V0),
                          (i64 GPR:$vl))),
            (!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)
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
    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "W",
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
    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "W",
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
    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "W",
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

let Predicates = [HasStdExtV] in {

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
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vaadd", "PseudoVAADD">;
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

foreach vti = NoGroupIntegerVectors in
  def : Pat<(vti.Vector (int_epi_vmv_s_x GPR:$rs2, GPR:$vl)),
            (PseudoVMV_S_X $rs2, (NoX0 GPR:$vl), vti.SEW)>;

defm "" : pat_intrinsic_binary_any_and_int_v_vx<"int_epi_vslide1up", "PseudoVSLIDE1UP">;
defm "" : pat_intrinsic_binary_any_and_int_v_vx<"int_epi_vslide1down", "PseudoVSLIDE1DOWN">;

foreach mti = AllMasks in {
  def : Pat<(int_epi_vmpopc (mti.Mask VR:$rs2), GPR:$vl),
            (!cast<Instruction>("PseudoVPOPC_M_M"#mti.VLMul) $rs2,
              (mti.Mask zero_reg), (NoX0 GPR:$vl), mti.SEW)>;
  def : Pat<(int_epi_vmpopc_mask (mti.Mask VR:$rs2), (mti.Mask V0), GPR:$vl),
            (!cast<Instruction>("PseudoVPOPC_M_M"#mti.VLMul) $rs2,
              (mti.Mask V0), (NoX0 GPR:$vl), mti.SEW)>;

  def : Pat<(int_epi_vmfirst (mti.Mask VR:$rs2), GPR:$vl),
            (!cast<Instruction>("PseudoVFIRST_M_M"#mti.VLMul) $rs2,
              (mti.Mask zero_reg), (NoX0 GPR:$vl), mti.SEW)>;
  def : Pat<(int_epi_vmfirst_mask (mti.Mask VR:$rs2), (mti.Mask V0), GPR:$vl),
            (!cast<Instruction>("PseudoVFIRST_M_M"#mti.VLMul) $rs2,
              (mti.Mask V0), (NoX0 GPR:$vl), mti.SEW)>;
}

//def : Pat<(int_epi_vmsbf VR:$rs1),
//        (VMSBF_M $rs1)>;
//def : Pat<(int_epi_vmsbf_mask VR:$rs1, V0),
//        (VMSBF_M_MASK $rs1, vmask_only_true.Value)>;

//def : Pat<(int_epi_vmsif VR:$rs1),
//        (VMSIF_M $rs1)>;
//def : Pat<(int_epi_vmsif_mask VR:$rs1, V0),
//        (VMSIF_M_MASK $rs1, vmask_only_true.Value)>;

//def : Pat<(int_epi_vmsof VR:$rs1),
//        (VMSOF_M $rs1)>;
//def : Pat<(int_epi_vmsof_mask VR:$rs1, V0),
//        (VMSOF_M_MASK $rs1, vmask_only_true.Value)>;

//foreach vtp = AllIntegerVectors in
//{
//def : Pat<(vtp.Vector (int_epi_viota VR:$rs1)),
//        (VIOTA_M $rs1)>;
//def : Pat<(vtp.Vector (int_epi_viota_mask VR:$rs1, V0)),
//        (VIOTA_M_MASK $rs1, vmask_only_true.Value)>;
//}

//foreach vtp = AllVectors in
//def : Pat<(vtp.Vector (int_epi_vcompress (vtp.Vector VR:$rs2), (nxv1i1 VR:$rs1))),
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
defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vmsub", "PseudoVNMSUB">;
defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vmacc", "PseudoVMACC">;
defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vmsac", "PseudoVNMSAC">;

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
             // generate the result in a register of class FPR32. When dealing
             // with the f64 variant of a pattern we need to promote the FPR32
             // subregister generated by the instruction to the FPR64 base
             // register expected by the type in the pattern
             !if(!eq(!cast<string>(fvti.ScalarRegClass),
                     !cast<string>(FPR64)),
                 (SUBREG_TO_REG (i32 -1),
                  (PseudoVFMV_F_S $rs2, (NoX0 GPR:$vl), fvti.SEW),
                  sub_32),
                 (PseudoVFMV_F_S $rs2, (NoX0 GPR:$vl), fvti.SEW))>;

  def : Pat<(fvti.Vector (int_epi_vfmv_s_f
                          (fvti.Scalar fvti.ScalarRegClass:$rs2), GPR:$vl)),
            (PseudoVFMV_S_F
             ToFPR32<fvti.Scalar, fvti.ScalarRegClass, "rs2">.ret,
             (NoX0 GPR:$vl), fvti.SEW)>;
}

defm "" : pat_intrinsic_binary_relational_fp_m_vv_vf<"int_epi_vmfeq", "PseudoVMFEQ">;
defm "" : pat_intrinsic_binary_relational_fp_m_vv_vf<"int_epi_vmfle", "PseudoVMFLE">;
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
            (!cast<Instruction>("PseudoVFSQRT_V_M" # vti.RegClass.VLMul)
             (vti.Vector zero_reg),
             $rs2,
             (vti.Mask zero_reg),
             (NoX0 GPR:$vl), vti.SEW)>;

foreach vti = AllFloatVectors in
  def : Pat<(vti.Vector (int_epi_vfsqrt_mask (vti.Vector vti.RegClass:$merge),
                         (vti.Vector vti.RegClass:$rs2), (vti.Mask V0),
                         GPR:$vl)),
            (!cast<Instruction>("PseudoVFSQRT_V_M" # vti.RegClass.VLMul)
             $merge,
             $rs2,
             (vti.Mask V0),
             (NoX0 GPR:$vl), vti.SEW)>;

//foreach itp = [ nxv1i32, nxv1i64 ] in
//foreach vtp = AllFloatVectors in
//def : Pat<(itp (int_epi_vfclass (vtp.Vector VR:$rs2))),
//          (VFCLASS_V $rs2)>;

//foreach itp = [ nxv1i32, nxv1i64 ] in
//foreach vtp = AllFloatVectors in
//def : Pat<(itp (int_epi_vfclass_mask (vtp.Vector VR:$rs2), V0)),
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
               (ivti.Vector zero_reg),
               $rs2,
               (ivti.Mask zero_reg),
               (NoX0 GPR:$vl), fvti.SEW)>;

    def : Pat<(ivti.Vector (!cast<Intrinsic>(intrinsic_float_to_int#"_mask")
                            (ivti.Vector fvti.RegClass:$merge),
                            (fvti.Vector fvti.RegClass:$rs2), (fvti.Mask V0),
                            GPR:$vl)),
              (!cast<Instruction>(instruction_float_to_int#"_M"#vlmul)
               $merge,
               $rs2,
               (fvti.Mask V0),
               (NoX0 GPR:$vl), fvti.SEW)>;

    def : Pat<(fvti.Vector (!cast<Intrinsic>(intrinsic_int_to_float)
                            (ivti.Vector fvti.RegClass:$rs2),
                            GPR:$vl)),
              (!cast<Instruction>(instruction_int_to_float#"_M"#vlmul)
               (fvti.Vector zero_reg),
               $rs2,
               (fvti.Mask zero_reg),
               (NoX0 GPR:$vl), fvti.SEW)>;

    def : Pat<(fvti.Vector (!cast<Intrinsic>(intrinsic_int_to_float#"_mask")
                            (fvti.Vector fvti.RegClass:$merge),
                            (ivti.Vector fvti.RegClass:$rs2), (fvti.Mask V0),
                            GPR:$vl)),
              (!cast<Instruction>(instruction_int_to_float#"_M"#vlmul)
               $merge,
               $rs2,
               (fvti.Mask V0),
               (NoX0 GPR:$vl), fvti.SEW)>;
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
               (wti.Vector zero_reg),
               vti.RegClass:$rs1,
               /* rs2 */ X0,
               (wti.Mask zero_reg),
               (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(wti.Vector (!cast<Intrinsic>(intrinsic_name#"_mask")
                           (wti.Vector wti.RegClass:$merge),
                           (vti.Vector vti.RegClass:$rs1),
                           (wti.Mask V0),
                           (i64 GPR:$vl))),
              (!cast<Instruction>(instruction_name#"_M"#vlmul) // FIXME change when fix nameing in binary intrinsics
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
               (vti.Vector zero_reg),
               wti.RegClass:$rs1,
               /* rs2 */ X0,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic_name#"_mask")
                           (vti.Vector vti.RegClass:$merge),
                           (wti.Vector wti.RegClass:$rs1),
                           (vti.Mask V0),
                           (i64 GPR:$vl))),
              (!cast<Instruction>(instruction_name#"_M"#vlmul) // FIXME change when fix nameing in binary intrinsics
               vti.RegClass:$merge,
               wti.RegClass:$rs1,
               /* rs2 */ X0,
               (vti.Mask V0),
               (NoX0 GPR:$vl), vti.SEW)>;
  }
}

defm : pat_intrinsic_binary_int_w_vX0<"int_epi_vwcvtu_x_x", "PseudoVWADDU_VX">;
defm : pat_intrinsic_binary_int_w_vX0<"int_epi_vwcvt_x_x", "PseudoVWADD_VX">;

defm : pat_intrinsic_binary_int_v_wX0<"int_epi_vncvt_x_x", "PseudoVNSRL_WX">;

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (int_epi_vid GPR:$vl)),
            (!cast<Instruction>("PseudoVID_V_M"#vti.RegClass.VLMul)
             (vti.Vector zero_reg),
             (vti.Mask zero_reg),
             (NoX0 GPR:$vl), vti.SEW)>;
  def : Pat<(vti.Vector (int_epi_vid_mask (vti.Vector vti.RegClass:$merge),
                         (vti.Mask V0), GPR:$vl)),
            (!cast<Instruction>("PseudoVID_V_M"#vti.RegClass.VLMul)
             $merge,
             (vti.Mask V0),
             (NoX0 GPR:$vl), vti.SEW)>;
}

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (int_epi_vbroadcast GPR:$rs2, GPR:$vl)),
            (!cast<Instruction>("PseudoVMV_V_X_M"#vti.RegClass.VLMul)
             $rs2,
             (NoX0 GPR:$vl), vti.SEW)>;
  def : Pat<(vti.Vector (int_epi_vbroadcast simm5:$imm5, GPR:$vl)),
            (!cast<Instruction>("PseudoVMV_V_I_M"#vti.RegClass.VLMul)
             simm5:$imm5,
             (NoX0 GPR:$vl), vti.SEW)>;
}
foreach fvti = AllFloatVectors in {
  def : Pat<(fvti.Vector (int_epi_vbroadcast
                         (fvti.Scalar fvti.ScalarRegClass:$rs2), GPR:$vl)),
            (!cast<Instruction>("PseudoVFMV_V_F_M"#fvti.RegClass.VLMul)
             ToFPR32<fvti.Scalar, fvti.ScalarRegClass, "rs2">.ret,
             (NoX0 GPR:$vl), fvti.SEW)>;
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
                          VReg result_reg_class,
                          VReg op_reg_class,
                          bit swap = 0>
{
  foreach instruction = [!cast<Instruction>(instruction_name#"_VV_M"#vlmul)] in
  def : Pat<(result_type (vop
                          (op_type op_reg_class:$rs1),
                          (op_type op_reg_class:$rs2))),
            swap_helper<
              (instruction (result_type zero_reg)),
              (instruction op_reg_class:$rs1),
              (instruction op_reg_class:$rs2),
              (instruction (mask_type zero_reg),
               VLMax,
               sew),
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
                           VReg result_reg_class,
                           VReg op1_reg_class,
                           VReg op2_reg_class> // FIXME Registerclass, dagoperand?
{
  def : Pat<(result_type (vop
                          (result_type result_reg_class:$rs3),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2))),
            (!cast<Instruction>(instruction_name#"_VV_M"#vlmul)
             result_reg_class:$rs3,
             ToFPR32<op1_type, op1_reg_class, "rs1">.ret,
             op2_reg_class:$rs2,
             (mask_type zero_reg),
             VLMax,
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

let Predicates = [HasStdExtV] in {

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
             ToFPR32<fvti.Scalar, fvti.ScalarRegClass, "value">.ret,
             VLMax,
             fvti.SEW)>;

  foreach ivti = [GetIntVectorTypeInfo<fvti>.Vti] in
  def : Pat<(fvti.Vector (sint_to_fp (ivti.Vector ivti.RegClass:$value))),
            (!cast<Instruction>("PseudoVFCVT_F_X_V_M" # fvti.RegClass.VLMul)
             (fvti.Vector zero_reg),
             $value,
             (ivti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;

  foreach ivti = [GetIntVectorTypeInfo<fvti>.Vti] in
  def : Pat<(fvti.Vector (uint_to_fp (ivti.Vector ivti.RegClass:$value))),
            (!cast<Instruction>("PseudoVFCVT_F_XU_V_M" # fvti.RegClass.VLMul)
             (fvti.Vector zero_reg),
             $value,
             (ivti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;

  foreach ivti = [GetIntVectorTypeInfo<fvti>.Vti] in
  def : Pat<(ivti.Vector (fp_to_sint (fvti.Vector fvti.RegClass:$value))),
            (!cast<Instruction>("PseudoVFCVT_X_F_V_M" # fvti.RegClass.VLMul)
             (ivti.Vector zero_reg),
             $value,
             (fvti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;

  foreach ivti = [GetIntVectorTypeInfo<fvti>.Vti] in
  def : Pat<(ivti.Vector (fp_to_uint (fvti.Vector fvti.RegClass:$value))),
            (!cast<Instruction>("PseudoVFCVT_XU_F_V_M" # fvti.RegClass.VLMul)
             (ivti.Vector zero_reg),
             $value,
             (fvti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;
}

foreach vti = AllIntegerVectors in
{
  def : Pat<(vti.Vector (riscv_vbroadcast GPR:$value)),
            (!cast<Instruction>("PseudoVMV_V_X_M"#vti.RegClass.VLMul)
             GPR:$value,
             VLMax, vti.SEW)>;
  def : Pat<(vti.Vector (riscv_vbroadcast simm5:$value)),
            (!cast<Instruction>("PseudoVMV_V_I_M"#vti.RegClass.VLMul)
             simm5:$value,
             VLMax, vti.SEW)>;

  def : Pat<(vti.Vector (int_experimental_vector_stepvector)),
            (!cast<Instruction>("PseudoVID_V_M"#vti.RegClass.VLMul)
             (vti.Vector zero_reg),
             (vti.Mask zero_reg),
             VLMax, vti.SEW)>;
}

// FIXME Replace VEXT patterns with VMV.X.S
//def : Pat<(extractelt (nxv1i64 EPIVR:$rs2), GPR:$rs1),
//          (PseudoVEXT_X_V (nxv1i64 EPIVR:$rs2), GPR:$rs1, 64)>;

//def : Pat<(extractelt (nxv2i64 EPIVR2:$rs2), GPR:$rs1),
//          (PseudoVEXT_X_V
//            (EXTRACT_SUBREG
//              (PseudoVSLIDEDOWN_VX_M2 (nxv2i64 EPIVR2:$rs2), GPR:$rs1, X0, 64),
//              epivreven),
//            X0, 64)>;

//def : Pat<(extractelt (nxv4i64 EPIVR4:$rs2), GPR:$rs1),
//          (PseudoVEXT_X_V
//            (EXTRACT_SUBREG
//              (PseudoVSLIDEDOWN_VX_M4 (nxv4i64 EPIVR4:$rs2), GPR:$rs1, X0, 64),
//              epivreven),
//            X0, 64)>;

//def : Pat<(extractelt (nxv8i64 EPIVR8:$rs2), GPR:$rs1),
//          (PseudoVEXT_X_V
//            (EXTRACT_SUBREG
//              (PseudoVSLIDEDOWN_VX_M8 (nxv8i64 EPIVR8:$rs2), GPR:$rs1, X0, 64),
//              epivreven),
//            X0, 64)>;

// def : Pat<(sext_inreg (extractelt (nxv2i32 EPIVR:$rs2), GPR:$rs1), i32),
//           (PseudoVEXT_X_V $rs2, $rs1, 64)>;
// def : Pat<(sext_inreg (extractelt (nxv4i16 EPIVR:$rs2), GPR:$rs1), i16),
//           (PseudoVEXT_X_V $rs2, $rs1, 64)>;
// def : Pat<(sext_inreg (extractelt (nxv8i16 EPIVR:$rs2), GPR:$rs1), i8),
//           (PseudoVEXT_X_V $rs2, $rs1, 64)>;

// FIXME: This should be done in a better way.
// FIXME This pattern is wrong
//def : Pat<(nxv1i64 (sext_inreg (nxv1i64 VR:$value), nxv1i32)),
//          (PseudoVSRA_VI_M1 (PseudoVSLL_VI_M1 $value, 32, X0, 64), 32, X0, 64)>;

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
              (!cast<Instruction>("PseudoVL"#width#"_V_M"#vlmul)
               (vti.Vector zero_reg),
               $rs1,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_strided GPR:$rs1, GPR:$rs2, GPR:$vl)),
              (!cast<Instruction>("PseudoVLS"#width#"_V_M"#vlmul)
               (vti.Vector zero_reg),
               $rs1, $rs2,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_indexed GPR:$rs1,
                           (ivti.Vector vti.RegClass:$rs2), GPR:$vl)),
              (!cast<Instruction>("PseudoVLX"#width#"_V_M"#vlmul)
               (vti.Vector zero_reg),
               $rs1, $rs2,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_mask (vti.Vector vti.RegClass:$merge),
                           GPR:$rs1, (vti.Mask V0), GPR:$vl)),
               (!cast<Instruction>("PseudoVL"#width#"_V_M"#vlmul)
                $merge,
                $rs1,
                (vti.Mask V0),
                (NoX0 GPR:$vl),
                vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_strided_mask
                           (vti.Vector vti.RegClass:$merge), GPR:$rs1, GPR:$rs2,
                           (vti.Mask V0), GPR:$vl)),
              (!cast<Instruction>("PseudoVLS"#width#"_V_M"#vlmul)
               $merge,
               $rs1, $rs2,
               (vti.Mask V0),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_indexed_mask
                           (vti.Vector vti.RegClass:$merge), GPR:$rs1,
                           (ivti.Vector vti.RegClass:$rs2), (vti.Mask V0),
                           GPR:$vl)),
              (!cast<Instruction>("PseudoVLX"#width#"_V_M"#vlmul)
               $merge,
               $rs1, $rs2,
               (vti.Mask V0),
               (NoX0 GPR:$vl),
               vti.SEW)>;
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
               $rs3, $rs1,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(int_epi_vstore_strided
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1, GPR:$rs2, GPR:$vl),
              (!cast<Instruction>("PseudoVSS"#width#"_V_M"#vlmul)
               $rs3, $rs1, $rs2,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(int_epi_vstore_indexed
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1,
               (ivti.Vector vti.RegClass:$rs2), GPR:$vl),
              (!cast<Instruction>("PseudoVSX"#width#"_V_M"#vlmul)
               $rs3, $rs1, $rs2,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(int_epi_vstore_mask
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1, (vti.Mask V0),
               GPR:$vl),
              (!cast<Instruction>("PseudoVS"#width#"_V_M"#vlmul)
               $rs3, $rs1,
               (vti.Mask V0),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(int_epi_vstore_strided_mask
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1, GPR:$rs2,
               (vti.Mask V0), GPR:$vl),
              (!cast<Instruction>("PseudoVSS"#width#"_V_M"#vlmul)
               $rs3, $rs1, $rs2,
               (vti.Mask V0),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(int_epi_vstore_indexed_mask
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1,
               (ivti.Vector vti.RegClass:$rs2), (vti.Mask V0), GPR:$vl),
              (!cast<Instruction>("PseudoVSX"#width#"_V_M"#vlmul)
               $rs3, $rs1, $rs2,
               (vti.Mask V0),
               (NoX0 GPR:$vl),
               vti.SEW)>;
  }
}

let Predicates = [HasStdExtV] in {

defm "" : pat_load<"E">;
defm "" : pat_store<"E">;

}
