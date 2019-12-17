//==-- epi-builtins.def - EPI RISCV-V Builtin function database --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file documents the EPI builtins.
//
//===----------------------------------------------------------------------===//

include "epi_builtins.td"

class DocEPIBuiltinCategory {
  string Name = "";
  int Index = 1000;
}

def DocEPICategoryNone : DocEPIBuiltinCategory;

class DocEPIBuiltin<EPIBuiltin builtin>
{
  EPIBuiltin Builtin = builtin;
  DocEPIBuiltinCategory Category = DocEPICategoryNone;
  string Title = "";
  string Description = "";
  string Instruction = "";
  string Operation = "";
  string OperationMask = "";
  list<string> ParameterNames = [];

  bit Undocumented = 0;
}

def DocEPIVectorConfiguration : DocEPIBuiltinCategory
{
  let Name = "Vector configuration";
  let Index = 10;
}

def DocEPIFloatingPointArithmetic : DocEPIBuiltinCategory
{
  let Name = "Floating-point arithmetic operations";
  let Index = 20;
}

def DocEPIFloatingPointRelational : DocEPIBuiltinCategory
{
  let Name = "Floating-point relational operations";
  let Index = 21;
}

def DocEPIIntegerArithmetic : DocEPIBuiltinCategory
{
  let Name = "Integer arithmetic operations";
  let Index = 30;
}

def DocEPIIntegerRelational : DocEPIBuiltinCategory
{
  let Name = "Integer relational operations";
  let Index = 31;
}

def DocEPIMemory : DocEPIBuiltinCategory
{
  let Name = "Memory accesses";
  let Index = 40;
}

def DocEPILaneManipulation : DocEPIBuiltinCategory
{
  let Name = "Vector elements manipulation";
  let Index = 50;
}

def DocEPIBitManipulation : DocEPIBuiltinCategory
{
  let Name = "Bit manipulation";
}

def DocEPIMaskConversions : DocEPIBuiltinCategory
{
  let Name = "Conversions between mask and integer vectors";
}

def DocEPIIntegerConversions : DocEPIBuiltinCategory
{
  let Name = "Conversions between integer vectors";
}

def DocEPIFloatConversions : DocEPIBuiltinCategory
{
  let Name = "Conversions between floating-point vectors";
}

def DocEPIIntFloatConversions : DocEPIBuiltinCategory
{
  let Name = "Conversions between integer and floating-point vectors";
}

def DocEPIMaskOperations : DocEPIBuiltinCategory
{
  let Name = "Operations with masks";
}


// FIXME: These will need to be documented but they are too niche
// for us.
let Undocumented = 1 in
{
  def : DocEPIBuiltin<vaadd>;
  def : DocEPIBuiltin<vadc>;
  def : DocEPIBuiltin<vasub>;
  def : DocEPIBuiltin<vdot>;
  def : DocEPIBuiltin<vdotu>;
  def : DocEPIBuiltin<vfdot>;
  def : DocEPIBuiltin<vmacc>;
  def : DocEPIBuiltin<vmadc>;
  def : DocEPIBuiltin<vmadd>;
  def : DocEPIBuiltin<vmsac>; // This one does not seem to exist anymore?
  def : DocEPIBuiltin<vmsbc>;
  def : DocEPIBuiltin<vmsub>; // This one does not seem to exist anymore?
  def : DocEPIBuiltin<vnclip>;
  def : DocEPIBuiltin<vnclipu>;
  def : DocEPIBuiltin<vrsub>;
  def : DocEPIBuiltin<vsadd>;
  def : DocEPIBuiltin<vsaddu>;
  def : DocEPIBuiltin<vsbc>;
  def : DocEPIBuiltin<vsmul>;
  def : DocEPIBuiltin<vssra>;
  def : DocEPIBuiltin<vssrl>;
  def : DocEPIBuiltin<vssub>;
  def : DocEPIBuiltin<vssubu>;
  def : DocEPIBuiltin<vwmacc>;
  def : DocEPIBuiltin<vwmaccu>;
  def : DocEPIBuiltin<vwmsac>;
  def : DocEPIBuiltin<vwmsacu>;
  def : DocEPIBuiltin<vwsmacc>;
  def : DocEPIBuiltin<vwsmaccu>;
  def : DocEPIBuiltin<vwsmsac>;
  def : DocEPIBuiltin<vwsmsacu>;
}

def : DocEPIBuiltin<cast_to_mask>
{
  let Category = DocEPIMaskConversions;
  let Title = "Reinterpret an integer vector as a vector mask";
  let Description = [{Use these builtins when you need to reinterpret the
contents of an integer vector as a vector mask. These builtins are a no-op and
exist only to transform the vector types.}];
}

def : DocEPIBuiltin<cast_from_mask>
{
  let Category = DocEPIMaskConversions;
  let Title = "Reinterpret an integer vector as a vector mask";
  let Description = [{Use these builtins when you need to reinterpret the
contents of an mask vector as a integer vector. These builtins are a no-op and
exist only to transform the vector types.}];
}

def : DocEPIBuiltin<vadd>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise integer addition";
  let Description = [{Use these builtins to do an elementwise addition of two
integer vectors.}];
  let Instruction = "vadd.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] + b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] + b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vsub>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise integer subtraction";
  let Description = [{Use these builtins to do an elementwise subtraction of two
integer vectors.}];
  let Instruction = "vsub.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] - b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] - b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vand>
{
  let Category = DocEPIBitManipulation;
  let Title = "Elementwise bitwise-and";
  let Description = [{Use these builtins to do an elementwise bitwise and of
two integer vectors.}];
  let Instruction = "vand.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = bitwise_and(a[element], b[element])
    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = bitwise_and(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vor>
{
  let Category = DocEPIBitManipulation;
  let Title = "Elementwise bitwise-or";
  let Description = [{Use these builtins to do an elementwise bitwise-or of
two integer vectors.}];
  let Instruction = "vor.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = bitwise_or(a[element], b[element])
    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = bitwise_or(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vbroadcast>
{
  let Category = DocEPILaneManipulation;
  let Title = "Broadcast a scalar value to all the elements of a vector";
  let Description = [{Use these builtins to create a vector where all the
elements have the value of a given scalar.}];
  let Instruction = "vmv.vi / vfmv.vx";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vcompress>
{
  let Category = DocEPILaneManipulation;
  let Title = "Pack elements contiguously";
  let Description = [{Use these builtins to create a vector where elements
selected by a mask vector are contiguosly packed.}];
  let Instruction = "vcompress.vm";

  let ParameterNames = ["a", "mask"];

  let Operation = [{
    next_index = 0
    for element = 0 to gvl - 1
       if mask[element] then
          result[next_index] = a[element]
          next_index = next_index + 1

    result[next_index : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vdiv>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise integer division";
  let Description = [{Use these builtins to do an elementwise division of two
integer vectors.}];
  let Instruction = "vdiv.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] / b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] / b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vdivu>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise unsigned integer division";
  let Description = [{Use these builtins to do an elementwise unsigned division
of two integer vectors.}];
  let Instruction = "vdivu.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = divu(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = divu(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vextract>
{
  let Category = DocEPILaneManipulation;
  let Title = "Extract integer element";
  let Description = [{Use these builtins to extract a single element from an
integer vector.}];
  let Instruction = "vext.x.v";

  let ParameterNames = ["a", "idx"];

  let Operation = [{
    if idx < VLMAX then
      result = a[idx]
    else
      result = 0
  }];
}

def : DocEPIBuiltin<vfadd>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise floating-point addition";
  let Description = [{Use these builtins to do an elementwise addition of two
floating-point vectors.}];
  let Instruction = "vfadd.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] + b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] + b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}


def : DocEPIBuiltin<vfcvt_f_x>
{
  let Category = DocEPIIntFloatConversions;
  let Title = "Integer to floating-point conversion";
  let Description = [{Use these builtins to convert elementwise an integer
vector to a floating-point vector.}];
  let Instruction = "vfcvt.f.x.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = int_to_fp(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = int_to_fp(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfcvt_f_xu>
{
  let Category = DocEPIIntFloatConversions;
  let Title = "Integer to floating-point conversion";
  let Description = [{Use these builtins to convert elementwise an unsigned
integer vector to a floating-point vector.}];
  let Instruction = "vfcvt.f.xu.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = uint_to_fp(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = uint_to_fp(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfcvt_x_f>
{
  let Category = DocEPIIntFloatConversions;
  let Title = "Floating-point to integer conversion";
  let Description = [{Use these builtins to convert elementwise a
floating-point vector to an integer vector.}];
  let Instruction = "vfcvt.x.f.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = fp_to_int(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = fp_to_int(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfcvt_xu_f>
{
  let Category = DocEPIIntFloatConversions;
  let Title = "Integer to floating-point conversion";
  let Description = [{Use these builtins to convert elementwise a
floating-point vector to an unsigned integer vector where elements are interpreted as
unsigned integers.}];
  let Instruction = "vfcvt.xu.f.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = fp_to_uint(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = fp_to_uint(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfdiv>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise floating-point division";
  let Description = [{Use these builtins to do an elementwise division of two
floating-point vectors.}];
  let Instruction = "vfdiv.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] / b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] / b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfmacc>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point multiply and add (overwrite addend)";
  let Description = [{Use these builtins to do an elementwise floating-point
multiply and add.

There is semantically no difference to `__builtin_epi_vfmadd`
except for the instruction used which determines the destructive nature
of this instruction.

At low level, the parameter `c` will be located in a vector register that will
be overwritten by the `vfmacc` instruction.}];
  let Instruction = "vfmacc.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] * b[element] + c[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] * b[element] + c[element]
       else
         result[element] = c[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfmadd>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point multiply and add (overwrite multiplicand)";
  let Description = [{Use these builtins to do an elementwise floating-point
multiply and add.

There is semantically no difference to `__builtin_epi_vfmacc`
except for the instruction used which determines the destructive nature
of this instruction.

At low level, the parameter `a` will be located in a vector register that will
be overwritten by the `vfmadd` instruction.}];
  let Instruction = "vfmadd.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] * b[element] + c[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] * b[element] + c[element]
       else
         result[element] = a[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfmax>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise floating-point maximum";
  let Description = [{Use these builtins to compute elementwise the maximum of two floating-point vectors.}];
  let Instruction = "vfmax.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = max(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = max(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfmerge>
{
  let Category = DocEPILaneManipulation;
  let Title = "Elementwise floating-point merge";
  let Description = [{Use these builtins to merge two floating-point vectors using a mask vector}];
  let Instruction = "vfmerge.vv";

  let ParameterNames = ["a", "b", "merge"];

  let Operation = [{
    for element = 0 to VLMAX
       if mask[element] then
         result[element] = a[element]
       else
         result[element] = b[element]
  }];
}

def : DocEPIBuiltin<vfmin>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise floating-point minimum";
  let Description = [{Use these builtins to compute elementwise the minimum of two floating-point vectors.}];
  let Instruction = "vfmin.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = min(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = min(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfmsac>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point multiply and subtract (overwrite subtrahend)";
  let Description = [{Use these builtins to do an elementwise floating-point
multiply and subtract.

There is semantically no difference to `__builtin_epi_vfmsub`
except for the instruction used which determines the destructive nature
of this instruction.

At low level, the parameter `c` will be located in a vector register that will
be overwritten by the `vfmsac` instruction.}];
  let Instruction = "vfmsac.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] * b[element] - c[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] * b[element] - c[element]
       else
         result[element] = c[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfmsub>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point multiply and subtract (overwrite multiplicand)";
  let Description = [{Use these builtins to do an elementwise floating-point
multiply and subtract.

There is semantically no difference to `__builtin_epi_vfmsac`
except for the instruction used which determines the destructive nature
of this instruction.

At low level, the parameter `a` will be located in a vector register that will
be overwritten by the `vfmsub` instruction.}];
  let Instruction = "vfmsub.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] * b[element] - c[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] * b[element] - c[element]
       else
         result[element] = a[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfmul>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise floating-point multiplication";
  let Description = [{Use these builtins to do an elementwise multiplication of two floating-point vectors.}];
  let Instruction = "vfmul.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] * b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] * b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfncvt_f_x>
{
  let Category = DocEPIIntFloatConversions;
  let Title = "Elementwise narrowing integer to floating-point conversion";
  let Description = [{Use these builtins to convert elementwise an integer
vector to a vector of floating-point elements that are half the width of the
integer element.}];
  let Instruction = "vfncvt.f.x.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = int_to_narrow_fp(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = int_to_narrow_fp(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfncvt_f_xu>
{
  let Category = DocEPIIntFloatConversions;
  let Title = "Elementwise narrowing unsigned integer to floating-point conversion";
  let Description = [{Use these builtins to convert elementwise an integer
vector, where elements are interpreted as unsigned integers, to a vector of
floating-point elements that are half the width of the integer element.}];
  let Instruction = "vfncvt.f.xu.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = uint_to_narrow_fp(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = uint_to_narrow_fp(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfncvt_x_f>
{
  let Category = DocEPIIntFloatConversions;
  let Title = "Elementwise narrowing integer to floating-point conversion";
  let Description = [{Use these builtins to convert elementwise an
floating-point vector to a vector of integer elements that are half the width
of the floating-point elements.}];
  let Instruction = "vfncvt.x.f.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = fp_to_narrow_int(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = fp_to_narrow_int(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfncvt_xu_f>
{
  let Category = DocEPIIntFloatConversions;
  let Title = "Elementwise narrowing unsigned integer to floating-point conversion";
  let Description = [{Use these builtins to convert elementwise an
floating-point vector to a vector of integer elements, interpreted as unsigned,
that are half the width of the floating-point elements.}];
  let Instruction = "vfncvt.xu.f.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = fp_to_narrow_uint(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = fp_to_narrow_uint(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfncvt_f_f>
{
  let Category = DocEPIFloatConversions;
  let Title = "Elementwise narrowing floating-point conversion";
  let Description = [{Use these builtins to convert elementwise a floating-point
vector to another vector of floating-point elements that are half the width of the
source vector.}];
  let Instruction = "vfncvt.f.f.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = fp_to_narrow_fp(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = fp_to_narrow_fp(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfnmacc>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point negate multiply and add (overwrite addend)";
  let Description = [{Use these builtins to do an elementwise floating-point
negate multiply and add.

There is semantically no difference to `__builtin_epi_vfnmadd`
except for the instruction used which determines the destructive nature
of this instruction.

At low level, the parameter `c` will be located in a vector register that will
be overwritten by the `vfnmacc` instruction.}];
  let Instruction = "vfnmacc.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = -( a[element] * b[element] ) - c[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = -( a[element] * b[element] ) - c[element]
       else
         result[element] = c[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfnmadd>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point multiply and add (overwrite multiplicand)";
  let Description = [{Use these builtins to do an elementwise floating-point
multiply and add.

There is semantically no difference to `__builtin_epi_vfnmacc`
except for the instruction used which determines the destructive nature
of this instruction.

At low level, the parameter `a` will be located in a vector register that will
be overwritten by the `vfnmadd` instruction.}];
  let Instruction = "vfnmadd.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = -( a[element] * b[element] ) - c[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = -( a[element] * b[element] ) - c[element]
       else
         result[element] = a[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfnmsac>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point negate multiply and subtract (overwrite subtrahend)";
  let Description = [{Use these builtins to do an elementwise floating-point
multiply and subtract.

There is semantically no difference to `__builtin_epi_vfnmsub`
except for the instruction used which determines the destructive nature
of this instruction.

At low level, the parameter `c` will be located in a vector register that will
be overwritten by the `vfnmsac` instruction.}];
  let Instruction = "vfnmsac.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = -( a[element] * b[element] ) + c[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = -( a[element] * b[element] ) + c[element]
       else
         result[element] = c[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfnmsub>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point negate multiply and subtract (overwrite multiplicand)";
  let Description = [{Use these builtins to do an elementwise floating-point
multiply and subtract.

There is semantically no difference to `__builtin_epi_vfnmsac`
except for the instruction used which determines the destructive nature
of this instruction.

At low level, the parameter `a` will be located in a vector register that will
be overwritten by the `vfnmsub` instruction.}];
  let Instruction = "vfnmsub.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] * b[element] - c[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] * b[element] - c[element]
       else
         result[element] = a[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfredmax>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point compute maximum of vector";
  let Description = [{Use these builtins to compute the maximum element of a
floating-point vector. The initial maximum is taken from the first element of
the vector `b`}];
  let Instruction = "vfredmax.vs";

  let Operation = [{
    if gvl > 0:
      current_max = b[0]
      for element = 0 to gvl - 1
         current_max = max(current_max, a[element])

      result[0] = current_max
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_max = b[0]
      for element = 0 to gvl - 1
         if mask[element] then
           current_max = max(current_max, a[element])
         else
           result[element] = merge[element]

      result[0] = current_max
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfredmin>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point compute minimum of vector";
  let Description = [{Use these builtins to compute the minium element of a
floating-point vector. The initial minium is taken from the first element of
the vector `b`}];
  let Instruction = "vfredmin.vs";

  let Operation = [{
    if gvl > 0:
      current_min = b[0]
      for element = 0 to gvl - 1
         current_min = max(current_min, a[element])

      result[0] = current_min
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_min = b[0]
      for element = 0 to gvl - 1
         if mask[element] then
           current_min = max(current_min, a[element])
         else
           result[element] = merge[element]

      result[0] = current_min
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfredosum>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point ordered sum of vector";
  let Description = [{Use these builtins to compute the sum of all the elements
of a floating-point vector. The initial result of the sum is taken from the
first element of the vector `b`.

This operation preserves the order of the floating-point addition as described.
}];
  let Instruction = "vfredosum.vs";

  let Operation = [{
    if gvl > 0:
      current_sum = b[0]
      for element = 0 to gvl - 1
         current_sum = current_sum + a[element]

      result[0] = current_sum
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_sum = b[0]
      for element = 0 to gvl - 1
         if mask[element] then
           current_sum = current_sum + a[element]
         else
           result[element] = merge[element]

      result[0] = current_sum
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfredsum>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point ordered sum of vector";
  let Description = [{Use these builtins to compute the sum of all the elements
of a floating-point vector. The initial result of the sum is taken from the
first element of the vector `b`.

This builtin will compute the sum in any valid sequential order.
}];
  let Instruction = "vfredsum.vs";

  let Operation = [{
    if gvl > 0:
      current_sum = b[0]
      for element = 0 to gvl - 1
         current_sum = current_sum + a[element]

      result[0] = current_sum
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_sum = b[0]
      for element = 0 to gvl - 1
         if mask[element] then
           current_sum = current_sum + a[element]
         else
           result[element] = merge[element]

      result[0] = current_sum
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfsgnj>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise floating-point sign copy";
  let Description = [{Use these builtins to generate a vector of floating-point elements
whose magnitude is the same as the elements of the first operand but their sign
is taken from the elements of the second operand.}];
  let Instruction = "vfsgnj.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = fsgnj(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = fsgnj(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfsgnjn>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise floating-point inverted sign copy";
  let Description = [{Use these builtins to generate a vector of floating-point elements
whose magnitude is the same as the elements of the first operand but their sign
is the opposite sign of the corresponding element of the second operand.

This is useful to negate a vector of floating-point element using the same
vector for the two operands.
}];
  let Instruction = "vfsgnjn.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = fsgnjn(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = fsgnjn(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfsgnjx>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise floating-point XOR sign";
  let Description = [{Use these builtins to generate a vector of floating-point
elements whose magnitude is the same as the elements of the first operand but
their sign is the exclusive or of their original sign and the sign of the
corresponding element of the second vector.

A positive floating-point element has a sign of 0. A negative floating-point
element has a sign of 1.

This is useful to compute the absolute value of a vector of floating-point
elements. To do that use the same vector for the two operands.}];
  let Instruction = "vfsgnjx.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = fsgnjx(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = fsgnjx(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfsqrt>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise floating-point square-root";
  let Description = [{Use these builtins to compute the elementwise square
root of a given floating-point vector.}];
  let Instruction = "vfsqrt.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = sqrt(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = sqrt(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfsub>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise floating-point addition";
  let Description = [{Use these builtins to do an elementwise subtraction of two
floating-point vectors.}];
  let Instruction = "vfsub.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] - b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] - b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwadd>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise widening floating-point addition";
  let Description = [{Use these builtins to do an elementwise addition of two
floating-point vectors.

Before doing the addition, the elements of the two vectors are widened to
floating-point values with twice the number of bits as the original
elements.}];
  let Instruction = "vfwadd.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = wide_fp(a[element]) + wide_fp(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = wide_fp(a[element]) + wide_fp(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwadd_w>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise widening floating-point addition (second operand)";
  let Description = [{Use these builtins to do an elementwise addition of two
floating-point vectors.

Before doing the addition, the elements of the second vector operand are
widened to floating-point values with twice the number of bits as the original
elements.}];
  let Instruction = "vfwadd.wv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] + wide_fp(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] + wide_fp(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwcvt_f_x>
{
  let Category = DocEPIIntFloatConversions;
  let Title = "Elementwise widening integer to floating-point conversion";
  let Description = [{Use these builtins to convert elementwise an integer
vector to a vector of floating-point elements that are twice the width of the
integer element.}];
  let Instruction = "vfwcvt.f.x.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = int_to_wide_fp(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = int_to_wide_fp(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwcvt_f_xu>
{
  let Category = DocEPIIntFloatConversions;
  let Title = "Elementwise widening unsigned integer to floating-point conversion";
  let Description = [{Use these builtins to convert elementwise an integer
vector, where elements are interpreted as unsigned integers, to a vector of
floating-point elements that are twice the width of the integer element.}];
  let Instruction = "vfwcvt.f.xu.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = uint_to_wide_fp(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = uint_to_wide_fp(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwcvt_x_f>
{
  let Category = DocEPIIntFloatConversions;
  let Title = "Elementwise widening integer to floating-point conversion";
  let Description = [{Use these builtins to convert elementwise an
floating-point vector to a vector of integer elements that are twice the width
of the floating-point elements.}];
  let Instruction = "vfwcvt.x.f.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = fp_to_wide_int(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = fp_to_wide_int(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwcvt_xu_f>
{
  let Category = DocEPIIntFloatConversions;
  let Title = "Elementwise widening unsigned integer to floating-point conversion";
  let Description = [{Use these builtins to convert elementwise an
floating-point vector to a vector of integer elements, interpreted as unsigned,
that are twice the width of the floating-point elements.}];
  let Instruction = "vfwcvt.xu.f.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = fp_to_wide_uint(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = fp_to_wide_uint(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwcvt_f_f>
{
  let Category = DocEPIFloatConversions;
  let Title = "Elementwise widening floating-point conversion";
  let Description = [{Use these builtins to convert elementwise a floating-point
vector to another vector of floating-point elements that are twice the width of the
source vector.}];
  let Instruction = "vfwcvt.f.f.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = fp_to_wide_fp(a[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = fp_to_wide_fp(a[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwmul>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise widening floating-point multiplication";
  let Description = [{Use these builtins to do an elementwise multiplication
of two floating-point vectors. Before doing the multiplication, the elements of
the second vector operand are widened to floating-point values with twice the
number of bits as the original elements.}];
  let Instruction = "vfwmul.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = widen_fp(a[element]) * widen_fp(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = widen_fp(a[element]) * widen_fp(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwmacc>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point widening multiply and add";
  let Description = [{Use these builtins to do an elementwise floating-point
multiply and add.

The elements of the three vector operands are widened to floating-point values
with twice the number of bits as the original elements.
}];
  let Instruction = "vfwmacc.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = widen_fp(a[element]) * widen_fp(b[element]) + widen_fp(c[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = widen_fp(a[element]) * widen_fp(b[element]) + widen_fp(c[element])
       else
         result[element] = c[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwmsac>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point widening multiply and subtract";
  let Description = [{Use these builtins to do an elementwise floating-point
multiply and subtract.

The elements of the three vector operands are widened to floating-point values
with twice the number of bits as the original elements.
}];
  let Instruction = "vfwmsac.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = widen_fp(a[element]) * widen_fp(b[element]) - widen_fp(c[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = widen_fp(a[element]) * widen_fp(b[element]) - widen_fp(c[element])
       else
         result[element] = c[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwnmacc>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point widening negate multiply and add";
  let Description = [{Use these builtins to do an elementwise floating-point
negate multiply and add.

The elements of the three vector operands are widened to floating-point values
with twice the number of bits as the original elements.
}];
  let Instruction = "vfnmacc.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = -( widen_fp(a[element]) * widen_fp(b[element]) ) - widen_fp(c[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = -( widen_fp(a[element]) * widen_fp(b[element]) ) - widen_fp(c[element])
       else
         result[element] = c[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwnmsac>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point widening negate multiply and subtract";
  let Description = [{Use these builtins to do an elementwise floating-point
multiply and subtract.

The elements of the three vector operands are widened to floating-point values
with twice the number of bits as the original elements.
}];
  let Instruction = "vfwnmsac.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = -( widen_fp(a[element]) * widen_fp(b[element]) ) + widen_fp(c[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = -( widen_fp(a[element]) * widen_fp(b[element]) ) + widen_fp(c[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwredosum>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point ordered sum of vector";
  let Description = [{Use these builtins to compute the sum of all the elements
of a floating-point vector. The initial result of the sum is taken from the
first element of the vector `b`.

This operation preserves the order of the floating-point addition as described.

The elements of the two vector operands are widened to floating-point values
with twice the number of bits as the original elements.
}];
  let Instruction = "vfwredosum.vs";

  let Operation = [{
    if gvl > 0:
      current_sum = fp_widen(b[0])
      for element = 0 to gvl - 1
         current_sum = current_sum + fp_widen(a[element])

      result[0] = current_sum
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_sum = fp_widen(b[0])
      for element = 0 to gvl - 1
         if mask[element] then
           current_sum = current_sum + fp_widen(a[element])
         else
           result[element] = merge[element]

      result[0] = current_sum
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwredsum>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Floating-point ordered sum of vector";
  let Description = [{Use these builtins to compute the sum of all the elements
of a floating-point vector. The initial result of the sum is taken from the
first element of the vector `b`.

This builtin will compute the sum in any valid sequential order.
}];
  let Instruction = "vfwredsum.vs";

  let Operation = [{
    if gvl > 0:
      current_sum = fp_widen(b[0])
      for element = 0 to gvl - 1
         current_sum = current_sum + fp_widen(a[element])

      result[0] = current_sum
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_sum = fp_widen(b[0])
      for element = 0 to gvl - 1
         if mask[element] then
           current_sum = current_sum + fp_widen(a[element])
         else
           result[element] = merge[element]

      result[0] = current_sum
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwsub>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise floating-point addition";
  let Description = [{Use these builtins to do an elementwise subtraction of two
floating-point vectors.

Before doing the subtraction, the elements of the two vectors are widened to
floating-point values with twice the number of bits as the original
elements.}];
  let Instruction = "vfwsub.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = fp_widen(a[element]) - fp_widen(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = fp_widen(a[element]) - fp_widen(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vfwsub_w>
{
  let Category = DocEPIFloatingPointArithmetic;
  let Title = "Elementwise floating-point addition";
  let Description = [{Use these builtins to do an elementwise subtraction of two
floating-point vectors.

Before doing the subtraction, the elements of second vector operand are widened to
floating-point values with twice the number of bits as the original
elements.}];
  let Instruction = "vfwsub.wv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] - fp_widen(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] - fp_widen(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vgetfirst>
{
  let Category = DocEPILaneManipulation;
  let Title = "Extract first element of a floating-point vector";
  let Description = [{Use these builtins to extract the first element of a vector.

This is useful when the result of some operation, like a reduction, is found
in the first element of a vector.}];

  let Instruction = "vfmv.f.s";

  let Operation = [{
    if gvl > 0 then
       result = a[0];
  }];
}

def : DocEPIBuiltin<vid>
{
  let Category = DocEPILaneManipulation;
  let Title = "Compute index vector";
  let Description = [{Use these builtins to compute an index vector.

An index vector is useful for indexed loads and stores and for register gathers.
}];

  let Instruction = "vid.v";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = element
    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = element
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<viota>
{
  let Category = DocEPILaneManipulation;
  let Title = "Compute a prefix sum of a mask";
  let Description = [{Use these builtins to compute the prefix sum given a mask
vector. For the purpose of the sum, elements enabled by the mask count as one,
otherwise they count as zero.}];

  let Instruction = "viota.m";

  let Operation = [{
    prefix_sum = 0
    for element = 0 to gvl - 1
       result[element] = prefix_sum
       if a[element] then
          prefix_sum = prefix_sum + 1

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    prefix_sum = 0
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = prefix_sum
         if a[element] then
            prefix_sum = prefix_sum + 1
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vload>
{
  let Category = DocEPIMemory;
  let Title = "Load contiguous elements from memory into a vector";
  let Description = [{Use these builtins to load elements contiguous in memory
into a vector.}];

  let Instruction = "vle.v";

  let ParameterNames = ["address"];

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = load_element(address)
      address = address + SEW / 8

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = load_element(address)
         address = address + SEW / 8
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vload_indexed>
{
  let Category = DocEPIMemory;
  let Title = "Load elements from memory into a vector using an index vector";
  let Description = [{Use these builtins to load elements into a vector
using an index vector. This is commonly known as a gather operation.

The elements of the index vector are added as an offset of bytes to the
`address` parameter to yield the effective address that loads the element of
the vector.}];

  let Instruction = "vlx.v";

  let ParameterNames = ["address", "index"];

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = load_element(address + index[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1

    result[gvl : VLMAX] = 0
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = load_element(address + index[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vload_indexed_unsigned>
{
  let Category = DocEPIMemory;
  let Title = "Load unsigned integer elements from memory into a vector using an index vector";
  let Description = [{Use these builtins to load unsigned integer elements into
a vector using an index vector. This is commonly known as a gather operation.

The elements of the index vector are added as an offset of bytes to the
`address` parameter to yield the effective address that loads the element of
the vector.

Depending on the types involved, there may not be semantic difference between
these builtins and the corresponding `__builtin_epi_vload_indexed` builtins.
}];

  let Instruction = "vlx.v";

  let ParameterNames = ["address", "index"];

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = load_unsigned_element(address + index[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1

    result[gvl : VLMAX] = 0
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = load_unsigned_element(address + index[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vload_mask>
{
  let Category = DocEPIMemory;
  let Title = "Load elements of a mask vector";
  let Description = [{Use these builtins to load the elements of a mask vector.

All the elements of the vector, in groups of 8 bits, are loaded.}];

  let Instruction = "vle.v";

  let ParameterNames = ["address"];

  let Operation = [{
    for element = 0 to VLMAX
      result[element] = load_uint8(address)
      address = address + 1
  }];
}

def : DocEPIBuiltin<vload_strided>
{
  let Category = DocEPIMemory;
  let Title = "Load strided elements from memory into a vector";
  let Description = [{Use these builtins to load elements into a vector
that are separated in memory by a constant stride value, in bytes.

The stride value is repeatedly added as an offset to the `address` parameter to
yield the effective address that loads the element of the vector.}];

  let Instruction = "vls.v";

  let ParameterNames = ["address", "stride"];

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = load_element(address)
      address = address + stride

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = load_element(address)
       else
         result[element] = merge[element]
      address = address + stride

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vload_strided_unsigned>
{
  let Category = DocEPIMemory;
  let Title = "Load unsigned strided elements from memory into a vector";
  let Description = [{Use these builtins to load elements into a vector
that are separated in memory by a constant stride value, in bytes.

The stride value is repeatedly added as an offset to the `address` parameter to
yield the effective address that loads the element of the vector.

Depending on the types involved, there may not be semantic difference between
these builtins and the corresponding `__builtin_epi_vload_strided`.
}];

  let Instruction = "vls.v";

  let ParameterNames = ["address", "stride"];

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = load_unsigned_element(address)
      address = address + stride

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = load_unsigned_element(address)
       else
         result[element] = merge[element]
      address = address + stride

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vload_unsigned>
{
  let Category = DocEPIMemory;
  let Title = "Load unsigned contiguous elements from memory into a vector";
  let Description = [{Use these builtins to load elements contiguous in memory
into a vector.

Depending on the types involved, there may not be semantic difference between
these builtins and the corresponding `__builtin_epi_vload`.
}];

  let Instruction = "vle.v";

  let ParameterNames = ["address"];

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = load_unsigned_element(address)
      address = address + SEW / 8

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = load_unsigned_element(address)
         address = address + SEW / 8
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmand>
{
  let Category = DocEPIMaskOperations;
  let Title = "Compute elementwise logical and between two masks";
  let Description = [{Use these builtins to compute a new mask that enables
an element if and only if the two mask operands enable that element.
}];

  let Instruction = "vmand.mm";

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = logical_and(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmandnot>
{
  let Category = DocEPIMaskOperations;
  let Title = "Compute elementwise logical andnot between two masks";
  let Description = [{Use these builtins to compute a new mask that enables
an element if and only if the first mask operand enables the element and the second
mask operand does not enable the element.}];

  let Instruction = "vmandnot.mm";

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = logical_and(a[element], logical_not(b[element]))

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmax>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise integer maximum";
  let Description = [{Use these builtins to compute elementwise the maximum of
two integer vectors.}];
  let Instruction = "vmax.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = max(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = max(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmaxu>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise integer maximum";
  let Description = [{Use these builtins to compute elementwise the maximum of
two unsigned integer vectors.}];
  let Instruction = "vmaxu.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = maxu(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = maxu(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmerge>
{
  let Category = DocEPILaneManipulation;
  let Title = "Elementwise integer merge";
  let Description = [{Use these builtins to merge two integer vectors using a mask vector}];
  let Instruction = "vmerge.vv";

  let ParameterNames = ["a", "b", "merge"];

  let Operation = [{
    for element = 0 to VLMAX
       if mask[element] then
         result[element] = a[element]
       else
         result[element] = b[element]
  }];
}

def : DocEPIBuiltin<vmfeq>
{
  let Category = DocEPIFloatingPointRelational;
  let Title = "Compare elementwise two floating-point vectors for equality";
  let Description = [{Use these builtins to compare to floating-point vectors
for equality.

The result is a mask that enables the element if the the floating-point
comparison holds for that element.}];
  let Instruction = "vmfeq.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] == b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] == b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmfge>
{
  let Category = DocEPIFloatingPointRelational;
  let Title = "Compare elementwise two floating-point vectors for greater-or-equal";
  let Description = [{Use these builtins to evaluate if the elements of the
first floating-point vector are greater or equal than the corresponding
elements of the second floating-point vector.

The result is a mask that enables the element if the the floating-point
comparison holds for that element.}];
  let Instruction = "vmfge.vf";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] >= b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] >= b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmfgt>
{
  let Category = DocEPIFloatingPointRelational;
  let Title = "Compare elementwise two floating-point vectors for greater-than";
  let Description = [{Use these builtins to evaluate if the elements of the
first floating-point vector are greater than, but not equal to, the corresponding
elements of the second floating-point vector.

The result is a mask that enables the element if the the floating-point
comparison holds for that element.}];
  let Instruction = "vmfgt.vf";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] > b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] > b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmfirst>
{
  let Category = DocEPIMaskOperations;
  let Title = "Compute the index of the first enabled element";
  let Description = [{Use these builtins to compute the lowermost index
of all the enabled elements of a given mask vector.}];

  let Instruction = "vmfirst.m";

  let Operation = [{
    result = -1
    for element = 0 to gvl - 1
      if a[element] then
        result = element
        break
  }];

  let OperationMask = [{
    result = -1
    for element = 0 to gvl - 1
      if mask[element] and a[element] then
        result = element
        break
  }];
}

def : DocEPIBuiltin<vmfle>
{
  let Category = DocEPIFloatingPointRelational;
  let Title = "Compare elementwise two floating-point vectors for lower-or-equal";
  let Description = [{Use these builtins to evaluate if the elements of the
first floating-point vector are lower or equal than the corresponding
elements of the second floating-point vector.

The result is a mask that enables the element if the the floating-point
comparison holds for that element.}];
  let Instruction = "vmfle.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] <= b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] <= b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmflt>
{
  let Category = DocEPIFloatingPointRelational;
  let Title = "Compare elementwise two floating-point vectors for greater-than";
  let Description = [{Use these builtins to evaluate if the elements of the
first floating-point vector are lower than, but not equal to, the corresponding
elements of the second floating-point vector.

The result is a mask that enables the element if the the floating-point
comparison holds.}];
  let Instruction = "vmflt.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] < b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] < b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmfne>
{
  let Category = DocEPIFloatingPointRelational;
  let Title = "Compare elementwise two floating-point vectors for inequality";
  let Description = [{Use these builtins to compare to floating-point vectors
for inequality.

The result is a mask that enables the element if the the floating-point
comparison holds for that element.}];
  let Instruction = "vmfne.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] != b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] != b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmford>
{
  let Category = DocEPIFloatingPointRelational;
  let Title = "Compute elementwise if floating-point vectors are ordered floating-point values";
  let Description = [{Use these builtins to check if the elements of the first floating-point vector
are ordered with respect to the corresponding element of the second floating-point. Two floating-point
values are ordered if neither is NaN.

The result is a mask that enables the element if the the floating-point
comparison holds for that element.}];
  let Instruction = "vmfne.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = logical_not(logical_or(is_nan(a[element]),
                                                is_nan(b[element])))

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = logical_not(logical_or(is_nan(a[element]),
                                                  is_nan(b[element])))
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmin>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise integer minimum";
  let Description = [{Use these builtins to compute elementwise the minimum of
two integer vectors.}];
  let Instruction = "vmin.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = min(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = min(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vminu>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise integer minimum";
  let Description = [{Use these builtins to compute elementwise the minimum of
two unsigned integer vectors.}];
  let Instruction = "vminu.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = minu(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = minu(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmnand>
{
  let Category = DocEPIMaskOperations;
  let Title = "Compute elementwise logical negated and between two masks";
  let Description = [{Use these builtins to compute a new mask that enables
an element if and only if the first and second mask operand do not enable
the element.}];

  let Instruction = "vmnand.mm";

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = logical_not(logical_and(a[element], b[element]))

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmnor>
{
  let Category = DocEPIMaskOperations;
  let Title = "Compute elementwise logical negated and between two masks";
  let Description = [{Use these builtins to compute a new mask that enables an
element if and only if neither the first or the second mask operand enable the
element.}];

  let Instruction = "vmnor.mm";

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = logical_not(logical_or(a[element], b[element]))

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmxor>
{
  let Category = DocEPIMaskOperations;
  let Title = "Compute elementwise logical exclusive or between two masks";
  let Description = [{Use these builtins to compute a new mask that enables
an element if and only if the first mask operand and the second mask operand
do not enable or disable that element at the same time.}];

  let Instruction = "vmxor.mm";

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = logical_xor(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmxnor>
{
  let Category = DocEPIMaskOperations;
  let Title = "Compute elementwise logical negated exclusive or between two masks";
  let Description = [{Use these builtins to compute a new mask that enables
an element if and only if the first mask operand and the second mask operand
enable or disable that element at the same time.}];

  let Instruction = "vmxnor.mm";

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = logical_not(logical_xor(a[element], b[element]))

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmor>
{
  let Category = DocEPIMaskOperations;
  let Title = "Compute elementwise logical xor between two masks";
  let Description = [{Use these builtins to compute a new mask that enables
an element if either the first or the second (or both) mask operands
enable that element.}];

  let Instruction = "vmand.mm";

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = logical_or(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmornot>
{
  let Category = DocEPIMaskOperations;
  let Title = "Compute elementwise logical xor between two masks";
  let Description = [{Use these builtins to compute a new mask that enables
an element if either the first mask operand enables the element or the second
mask operand does not enable that element.}];

  let Instruction = "vmornot.mm";

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = logical_or(a[element], logical_not(b[element]))

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmpopc>
{
  let Category = DocEPIMaskOperations;
  let Title = "Population count of a mask vector";
  let Description = [{Use these builtins to count the number of elements
that are enabled by a mask.}];

  let Instruction = "vmpopc.m";

  let Operation = [{
    result = 0
    for element = 0 to gvl - 1
      if a[element] then
        result = result + 1
  }];

  let OperationMask = [{
    result = 0
    for element = 0 to gvl - 1
      if mask[element] and a[element] then
        result = result + 1
  }];
}

def : DocEPIBuiltin<vmsbf>
{
  let Category = DocEPIMaskOperations;
  let Title = "Enable elements before the first one enabled";
  let Description = [{Use these builtins to compute a mask vector given another
mask vector. The resulting mask vector will have all the elements enabled up to,
but not including, the first element enabled in the given mask.

Every other element in the resulting mask is disabled.
}];

  let Instruction = "vmsbf.m";

  let Operation = [{
    for element = 0 to gvl - 1
      if not a[element] then
        result[element] = 1
      else
        break

    result[element : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
      if mask[element] then
        if not a[element] then
          result[element] = 1
        else
          break

    result[element : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmsif>
{
  let Category = DocEPIMaskOperations;
  let Title = "Enable elements until the first one enabled";
  let Description = [{Use these builtins to compute a mask vector given another
mask vector. The resulting mask vector will have all the elements enabled up to
the first element enabled in the given mask. The enabled element is included.

Every other element in the resulting mask is disabled.
}];

  let Instruction = "vmsif.m";

  let Operation = [{
    for element = 0 to gvl - 1
      result[element] = 1
      if a[element] then
        break

    result[element : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
      if mask[element] then
        result[element] = 1
        if a[element] then
          break

    result[element : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmseq>
{
  let Category = DocEPIIntegerRelational;
  let Title = "Compare elementwise two integer vectors for equality";
  let Description = [{Use these builtins to compare to floating-point vectors
for equality.

The result is a mask that enables the element if the the floating-point
comparison holds for that element.}];
  let Instruction = "vmseq.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] == b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] == b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmsgt>
{
  let Category = DocEPIIntegerRelational;
  let Title = "Compare elementwise two integer vectors for greater-than";
  let Description = [{Use these builtins to evaluate if the elements of the
first integer vector are greater than, but not equal to, the corresponding
elements of the second integer vector.

The result is a mask that enables the element if the the integer
comparison holds for that element.}];
  let Instruction = "vmsgt.vx";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] > b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] > b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmsgtu>
{
  let Category = DocEPIIntegerRelational;
  let Title = "Compare elementwise two unsigned integer vectors for greater-than";
  let Description = [{Use these builtins to evaluate if the elements of the
first unsigned integer vector are greater than, but not equal to, the corresponding
elements of the second unsigned integer vector.

The result is a mask that enables the element if the the unsigned integer
comparison holds for that element.}];
  let Instruction = "vmsgtu.vx";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] > b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] > b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmsof>
{
  let Category = DocEPIMaskOperations;
  let Title = "Enable only the first element enabled";
  let Description = [{Use these builtins to compute a mask vector given another
mask vector. The resulting mask vector will have all the elements disabled
except for the lowermost corresponding element enabled in the given mask.
}];

  let Instruction = "vmsof.m";

  let Operation = [{
    for element = 0 to gvl - 1
      if a[element] then
        result[element] = 1
        break
      else
        result[element] = 0

    result[element : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
      if mask[element] then
        if a[element] then
          result[element] = 1
          break
        else
          result[element] = 0

    result[element : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmslt>
{
  let Category = DocEPIIntegerRelational;
  let Title = "Compare elementwise two integer vectors for lower-than";
  let Description = [{Use these builtins to evaluate if the elements of the
first integer vector are lower than, but not equal to, the corresponding
elements of the second integer vector.

The result is a mask that enables the element if the the integer
comparison holds for that element.}];
  let Instruction = "vmslt.vx";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] < b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] < b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmsltu>
{
  let Category = DocEPIIntegerRelational;
  let Title = "Compare elementwise two unsigned integer vectors for lower-than";
  let Description = [{Use these builtins to evaluate if the elements of the
first unsigned integer vector are lower than, but not equal to, the corresponding
elements of the second unsigned integer vector.

The result is a mask that enables the element if the the unsigned integer
comparison holds for that element.}];
  let Instruction = "vmsltu.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] < b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] < b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmsle>
{
  let Category = DocEPIIntegerRelational;
  let Title = "Compare elementwise two integer vectors for lower-than-or-equal";
  let Description = [{Use these builtins to evaluate if the elements of the
first integer vector are lower than or equal to the corresponding
elements of the second integer vector.

The result is a mask that enables the element if the the integer
comparison holds for that element.}];
  let Instruction = "vmsle.vx";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] <= b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] <= b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmsleu>
{
  let Category = DocEPIIntegerRelational;
  let Title = "Compare elementwise two unsigned integer vectors for lower-than-or-equal";
  let Description = [{Use these builtins to evaluate if the elements of the
first unsigned integer vector are lower than or equal to the corresponding
elements of the second unsigned integer vector.

The result is a mask that enables the element if the the unsigned integer
comparison holds for that element.}];
  let Instruction = "vmsleu.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] <= b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] <= b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmsne>
{
  let Category = DocEPIIntegerRelational;
  let Title = "Compare elementwise two integer vectors for inequality";
  let Description = [{Use these builtins to compare to floating-point vectors
for inequality.

The result is a mask that enables the element if the the floating-point
comparison holds for that element.}];
  let Instruction = "vmsne.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] != b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] != b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmul>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise integer multiplication";
  let Description = [{Use these builtins to do an elementwise multiplication
of two integer vectors.

This operation returns the lower bits of the multiplication}];
  let Instruction = "vmul.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] * b[element]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] * b[element]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmulh>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise integer multiplication (higher bits)";
  let Description = [{Use these builtins to do an elementwise multiplication
of two integer vectors.

This operation returns the higher bits of the multiplication}];
  let Instruction = "vmulh.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = mulh(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = mulh(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmulhu>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise unsigned integer multiplication (higher bits)";
  let Description = [{Use these builtins to do an elementwise multiplication
of two unsigned integer vectors.

This operation returns the higher bits of the multiplication}];
  let Instruction = "vmulhu.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = mulhu(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = mulhu(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vmulhsu>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise mixed sign integer multiplication (higher bits)";
  let Description = [{Use these builtins to do an elementwise multiplication
of two unsigned integer vectors.

The second integer operand is interpreted as an unsigned integer vector.

This operation returns the higher bits of the multiplication}];
  let Instruction = "vmulhsu.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = mulhsu([element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = mulhsu([element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vnsrl>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Narrowing elementwise logical shift-right";
  let Description = [{Use these builtins to do a logical shift right of the
elements of the first operand with the shift amount specified in the elements
of the second operand.

The result of the operation is a vector of integer elements whose bitwidth is
half of the input integer vector operands.

This builtin can be used to narrow an integer vector if the shift amount is set
to zero.}];
  let Instruction = "vnsrl.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = narrow_int(srl(a[element], b[element]))

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = narrow_int(srl(a[element], b[element]))
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vnsra>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Narrowing elementwise arithmetic shift-right";
  let Description = [{Use these builtins to do an arithmetic shift-right of the
elements of the first operand with the shift amount specified in the elements
of the second operand.

The result of the operation is a vector of integer elements whose bitwidth is
half of the input integer vector operands.

This builtin can be used to narrow an integer vector if the shift amount is set
to zero.}];
  let Instruction = "vnsra.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = narrow_int(sra(a[element], b[element]))

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = narrow_int(sra(a[element], b[element]))
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vredand>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Integer vector bitwise-and reduction";
  let Description = [{Use these builtins to compute the bitwise-and of all the elements
of a integer vector. The initial result of the bitwise-and is taken from the
first element of the vector `b`.
}];
  let Instruction = "vredand.vs";

  let Operation = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         current_red = bitwise_and(current_red, a[element])

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         if mask[element] then
           current_red = bitwise_and(current_red, a[element])
         else
           result[element] = merge[element]

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vredmax>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Integer vector maximum reduction";
  let Description = [{Use these builtins to compute the maximum of all the
elements of a integer vector and a given scalar value. The scalar value is
taken from the first element of the vector `b`.
}];
  let Instruction = "vredmax.vs";

  let Operation = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         current_red = max(current_red, a[element])

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         if mask[element] then
           current_red = max(current_red, a[element])
         else
           result[element] = merge[element]

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vredmaxu>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Unsigned integer vector maximum reduction";
  let Description = [{Use these builtins to compute the maximum of all the
elements of a integer vector and a given scalar value. The scalar value is
taken from the first element of the vector `b`.
}];
  let Instruction = "vredmaxu.vs";

  let Operation = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         current_red = maxu(current_red, a[element])

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         if mask[element] then
           current_red = maxu(current_red, a[element])
         else
           result[element] = merge[element]

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vredmin>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Integer vector minimum reduction";
  let Description = [{Use these builtins to compute the minimum of all the
elements of a integer vector and a given scalar value. The scalar value is
taken from the first element of the vector `b`.
}];
  let Instruction = "vredmin.vs";

  let Operation = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         current_red = min(current_red, a[element])

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         if mask[element] then
           current_red = min(current_red, a[element])
         else
           result[element] = merge[element]

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vredminu>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Unsigned integer vector minimum reduction";
  let Description = [{Use these builtins to compute the minimum of all the
elements of a integer vector and a given scalar value. The scalar value is
taken from the first element of the vector `b`.
}];
  let Instruction = "vredminu.vs";

  let Operation = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         current_red = minu(current_red, a[element])

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         if mask[element] then
           current_red = minu(current_red, a[element])
         else
           result[element] = merge[element]

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vredor>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Integer vector bitwise-or reduction";
  let Description = [{Use these builtins to compute the bitwise-or of all the elements
of a integer vector. The initial result of the bitwise-or is taken from the
first element of the vector `b`.
}];
  let Instruction = "vredor.vs";

  let Operation = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         current_red = bitwise_or(current_red, a[element])

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         if mask[element] then
           current_red = bitwise_or(current_red, a[element])
         else
           result[element] = merge[element]

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vredsum>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Sum of integer vector";
  let Description = [{Use these builtins to compute the sum of all the elements
of an integer vector. The initial result of the sum is taken from the
first element of the vector `b`.
}];
  let Instruction = "vredsum.vs";

  let Operation = [{
    if gvl > 0:
      current_sum = b[0]
      for element = 0 to gvl - 1
         current_sum = current_sum + a[element]

      result[0] = current_sum
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_sum = b[0]
      for element = 0 to gvl - 1
         if mask[element] then
           current_sum = current_sum + a[element]
         else
           result[element] = merge[element]

      result[0] = current_sum
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vredxor>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Integer vector bitwise-xor reduction";
  let Description = [{Use these builtins to compute the bitwise-xor of all the elements
of a integer vector. The initial result of the bitwise-or is taken from the
first element of the vector `b`.
}];
  let Instruction = "vredxor.vs";

  let Operation = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         current_red = bitwise_xor(current_red, a[element])

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];

  let OperationMask = [{
    if gvl > 0:
      current_red = b[0]
      for element = 0 to gvl - 1
         if mask[element] then
           current_red = bitwise_xor(current_red, a[element])
         else
           result[element] = merge[element]

      result[0] = current_red
      result[1 : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vrem>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise integer division remainder";
  let Description = [{Use these builtins to compute the elementwise integer
division remainder of two integer vectors.}];
  let Instruction = "vrem.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = rem(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = rem(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vremu>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise unsigned integer division remainder";
  let Description = [{Use these builtins to compute the elementwise integer
division remainder of two unsigned integer vectors.}];
  let Instruction = "vremu.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = remu(a[element], b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = remu(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vrgather>
{
  let Category = DocEPILaneManipulation;
  let Title = "Register gather";
  let Description = [{Use these builtins to permute the elements of a vector
based on a vector of indices.}];
  let Instruction = "vrgather.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       if b[element] > VLMAX then
         result[element] = 0
       else
         result[element] = a[b[element]]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         if b[element] > VLMAX then
           result[element] = 0
         else
           result[element] = a[b[element]]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vsplat>
{
  let Category = DocEPILaneManipulation;
  let Title = "Broadcast the element of a vector to all the elements of a vector";
  let Description = [{Use these builtins to create a vector where all the
elements have the value of an element of another vector}];
  let Instruction = "vrgather.vx / vrgather.vi";

  let Operation = [{
    for element = 0 to gvl - 1
       if b > VLMAX then
         result[element] = 0
       else
         result[element] = a[b]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         if b > VLMAX then
           result[element] = 0
         else
           result[element] = a[b]
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vsetfirst_int>
{
  let Category = DocEPILaneManipulation;
  let Title = "Set first integer element of integer vector";
  let Description = [{Use these builtins to set the first element of a vector
to a given value.}];
  let Instruction = "vmv.s.x";

  let Operation = [{
     result[0] = b
     result[1 : gvl - 1]= a[1:gvl]
     result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vsetfirst_float>
{
  let Category = DocEPILaneManipulation;
  let Title = "Set first floating-point element of floating-point vector";
  let Description = [{Use these builtins to set the first element of a vector
to a given value.}];
  let Instruction = "vfmv.s.f";

  let Operation = [{
     result[0] = b
     result[1 : gvl - 1]= a[1:gvl]
     result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vsetvl>
{
  let Category = DocEPIVectorConfiguration;
  let Title = "Change the granted vector length";
  let Description = [{Use this builtin to set the granted vector length given a requested
vector length (`rvl`), and single element width (`sew`) and a length multiplier (`lmul`).

This builtin returns the granted vector length. This granted vector length is
suitable to use in other builtins that require it.

Valid values for the `sew` parameter are:

- `__epi_e8`, for elements of 8 bits (like `char`, `signed char`, `unsigned char`)
- `__epi_e16`, for elements of 16 bits (like `short`, `unsigned short` or the unsupported `__float16`)
- `__epi_e32`, for elements of 32 bits (like `int`, `unsigned int` or `float`)
- `__epi_e64`, for elements of 64 bit (like `long`, `unsigned long` or `double`)

Valid values for the `lmul` parameter are:

- `__epi_m1`, for `LMUL=1`
- `__epi_m2`, for `LMUL=2`
- `__epi_m4`, for `LMUL=4`
- `__epi_m8`, for `LMUL=8`
}];
  let Instruction = "vsetvli";

  let ParameterNames = ["rvl", "sew", "lmul"];

  let Operation = [{
    gvl = compute_vector_length(rvl, sew, lmul)
    result = gvl
  }];
}

def : DocEPIBuiltin<vsetvlmax>
{
  let Category = DocEPIVectorConfiguration;
  let Title = "Set the granted vector length to the maximum length";
  let Description = [{Use this builtin to set the granted vector length to the maximum value
allowed given a single element width (`sew`) and a length multiplier (`lmul`).

This builtin returns the granted vector length. This granted vector length is suitable
to use in other builtins that require it.}];
  let Instruction = "vsetvli";

  let ParameterNames = ["sew", "lmul"];

  let Operation = [{
    gvl = compute_vlmax(sew, lmul)
    result = gvl
  }];
}

def : DocEPIBuiltin<vslidedown>
{
  let Category = DocEPILaneManipulation;
  let Title = "Slide down elements of a vector";
  let Description = [{Use these builtins to "slide down" the elements of a
vector.}];

  let Instruction = "vslidedown.vx";

  let ParameterNames = ["a", "offset"];

  let Operation = [{
    for element = 0 to gvl - 1
      if element + offset < VLMAX then
         result[element] = a[element + offset]
      else
         result[element] = 0

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
      if mask[element] then
        if element + offset < VLMAX then
           result[element] = a[element + offset]
        else
           result[element] = 0
      else
        result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vslideup>
{
  let Category = DocEPILaneManipulation;
  let Title = "Slide up elements of a vector";
  let Description = [{Use these builtins to "slide up" the elements of a
vector.}];

  let Instruction = "vslideup.vx";

  let ParameterNames = ["a", "offset"];

  let Operation = [{
    for element = offset to gvl - 1
      result[i] = a[element - offset]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vslide1down>
{
  let Category = DocEPILaneManipulation;
  let Title = "Slide down elements of a vector one position";
  let Description = [{Use these builtins to "slide down" the elements of a
vector one position.}];

  let Instruction = "vslide1down.vx";

  let ParameterNames = ["a", "value"];

  let Operation = [{
    for element = 0 to gvl - 2
         result[element] = a[element + 1]
    result[gvl - 1] = value

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
      if mask[element] then
        if element < gvl - 1 then
           result[element] = a[element + 1]
        else
           result[element] = value
      else
        result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vslide1up>
{
  let Category = DocEPILaneManipulation;
  let Title = "Slide up elements of a vector one position";
  let Description = [{Use these builtins to "slide up" the elements of a
vector one position.}];

  let Instruction = "vslide1up.vx";

  let ParameterNames = ["a", "value"];

  let Operation = [{
    result[0] = value
    for element = 1 to gvl - 1
      result[i] = a[element - offset]

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
      if mask[element] then
        if element == 0 then
          result[0] = value
        else
          result[i] = a[element - offset]
      else
        result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vsll>
{
  let Category = DocEPIBitManipulation;
  let Title = "Elementwise logical shift left";
  let Description = [{Use these builtins compute the elementwise logical shift
left given two integer vector operands.}];
  let Instruction = "vsll.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = sll(a[element], b[element])
    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = sll(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vsrl>
{
  let Category = DocEPIBitManipulation;
  let Title = "Elementwise logical shift right";
  let Description = [{Use these builtins compute the elementwise logical shift
right given two integer vector operands.}];
  let Instruction = "vsrl.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = srl(a[element], b[element])
    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = srl(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vsra>
{
  let Category = DocEPIBitManipulation;
  let Title = "Elementwise arithmetic shift right";
  let Description = [{Use these builtins compute the elementwise arithmetic shift
right given two integer vector operands.}];
  let Instruction = "vsra.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = sra(a[element], b[element])
    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = sra(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vstore>
{
  let Category = DocEPIMemory;
  let Title = "Store vector elements into contiguous locations in memory";
  let Description = [{Use these builtins to store the elements of a vector into
contiguous locations in memory.}];

  let Instruction = "vse.v";

  let ParameterNames = ["address", "value"];

  let Operation = [{
    for element = 0 to gvl - 1
      store_element(address, value[element])
      address = address + SEW / 8
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
      if mask[element] then
        store_element(address, value[element])
        address = address + SEW / 8
  }];
}

def : DocEPIBuiltin<vstore_unsigned>
{
  let Category = DocEPIMemory;
  let Title = "Store unsigned vector elements into contiguous locations in memory";
  let Description = [{Use these builtins to store the elements of a vector into
contiguous locations in memory.}];

  let Instruction = "vse.v";

  let ParameterNames = ["address", "value"];

  let Operation = [{
    for element = 0 to gvl - 1
      store_element(address, value[element])
      address = address + SEW / 8
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
      if mask[element] then
        store_element(address, value[element])
        address = address + SEW / 8
  }];
}

def : DocEPIBuiltin<vstore_indexed>
{
  let Category = DocEPIMemory;
  let Title = "Store vector elements into memory using an index vector";
  let Description = [{Use these builtins to store the elements of a vector into
memory using an index vector. This is commonly known as a scatter operation.

The elements of the index vector are added as an offset of bytes to the
`address` parameter to yield the effective address that loads the element of
the vector.}];

  let Instruction = "vsx.v";

  let ParameterNames = ["address", "value", "indexes"];

  let Operation = [{
    for element = 0 to gvl - 1
      store_element(address + index[element], value[element])
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
      if mask[element] then
        store_element(address + index[element], value[element])
  }];
}

def : DocEPIBuiltin<vstore_indexed_unsigned>
{
  let Category = DocEPIMemory;
  let Title = "Store unsigned vector elements into memory using an index vector";
  let Description = [{Use these builtins to store the elements of a vector into
memory using an index vector. This is commonly known as a scatter operation.

The elements of the index vector are added as an offset of bytes to the
`address` parameter to yield the effective address that loads the element of
the vector.}];

  let Instruction = "vsx.v";

  let ParameterNames = ["address", "value", "indexes"];

  let Operation = [{
    for element = 0 to gvl - 1
      store_element(address + index[element], value[element])
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
      if mask[element] then
        store_element(address + index[element], value[element])
  }];
}

def : DocEPIBuiltin<vstore_mask>
{
  let Category = DocEPIMemory;
  let Title = "Store elements of a mask vector into memory";
  let Description = [{Use these builtins to store the elements of a mask vector into memory.

All the elements of the vector, in groups of 8 bits, are stored.}];

  let Instruction = "vse.v";

  let ParameterNames = ["address", "value"];

  let Operation = [{
    for element = 0 to VLMAX
      store_uint8(address, result[element])
      address = address + 1
  }];
}

def : DocEPIBuiltin<vstore_strided>
{
  let Category = DocEPIMemory;
  let Title = "Store elements of a vector into strided locations in memory";
  let Description = [{Use these builtins to store elements from a vector
into memory locations separated by a constant stride value, in bytes.

The stride value is repeatedly added as an offset to the `address` parameter to
yield the effective address that loads the element of the vector.}];

  let Instruction = "vss.v";

  let ParameterNames = ["address", "value", "stride"];

  let Operation = [{
    for element = 0 to gvl - 1
      store_element(address, result[element])
      address = address + stride

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         store_element(address, result[element])
      address = address + stride

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vstore_strided_unsigned>
{
  let Category = DocEPIMemory;
  let Title = "Store unsigned elements of a vector into strided locations in memory";
  let Description = [{Use these builtins to store elements from a vector
into memory locations separated by a constant stride value, in bytes.

The stride value is repeatedly added as an offset to the `address` parameter to
yield the effective address that loads the element of the vector.}];

  let Instruction = "vss.v";

  let ParameterNames = ["address", "value", "stride"];

  let Operation = [{
    for element = 0 to gvl - 1
      store_element(address, result[element])
      address = address + stride

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         store_element(address, result[element])
      address = address + stride

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vxor>
{
  let Category = DocEPIBitManipulation;
  let Title = "Elementwise bitwise-or";
  let Description = [{Use these builtins to do an elementwise bitwise-xor of
two integer vectors}];
  let Instruction = "vxor.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = bitwise_xor(a[element], b[element])
    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = bitwise_xor(a[element], b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vwadd>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise widening integer addition";
  let Description = [{Use these builtins to do an elementwise addition of two
integer vectors.

Before doing the addition, the elements of the two vectors are widened to
integer values with twice the number of bits as the original
elements.}];
  let Instruction = "vwadd.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = wide_int(a[element]) + wide_int(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = wide_int(a[element]) + wide_int(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vwadd_w>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise widening integer addition (second operand)";
  let Description = [{Use these builtins to do an elementwise addition of two
integer vectors.

Before doing the addition, the elements of the second vector operand are
widened to integer values with twice the number of bits as the original
elements.}];
  let Instruction = "vwadd.wv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] + wide_int(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] + wide_int(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vwaddu>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise widening unsigned integer addition";
  let Description = [{Use these builtins to do an elementwise addition of two
integer vectors.

Before doing the addition, the elements of the two vectors are widened to
integer values with twice the number of bits as the original
elements.}];
  let Instruction = "vwaddu.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = wide_int(a[element]) + wide_int(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = wide_int(a[element]) + wide_int(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vwaddu_w>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise widening unsigned integer addition (second operand)";
  let Description = [{Use these builtins to do an elementwise addition of two
integer vectors.

Before doing the addition, the elements of the second vector operand are
widened to integer values with twice the number of bits as the original
elements.}];
  let Instruction = "vwaddu.wv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] + wide_uint(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] + wide_uint(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}


def : DocEPIBuiltin<vwmul>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise widening integer multiplication";
  let Description = [{Use these builtins to do an elementwise multiplication
of two integer vectors.

This operation returns the lower bits of the multiplication}];
  let Instruction = "vwmul.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = widen_int(a[element]) * widen_int(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = widen_int(a[element]) * widen_int(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vwmulu>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise widening unsigned integer multiplication";
  let Description = [{Use these builtins to do an elementwise multiplication
of two integer vectors.

This operation returns the higher bits of the multiplication}];
  let Instruction = "vwmulu.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = widen_uint(a[element]) * widen_uint(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = widen_uint(a[element]) * widen_uint(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vwmulsu>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise widening integer multiplication (mixed signs)";
  let Description = [{Use these builtins to do an elementwise multiplication
of two integer vectors.

This operation returns the higher bits of the multiplication}];
  let Instruction = "vwmulsu.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = widen_int(a[element]) * widen_uint(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = widen_int(a[element]) * widen_uint(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vwsub>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise widening integer subtraction";
  let Description = [{Use these builtins to do an elementwise subtraction of two
integer vectors.

Before doing the subtraction, the elements of the two vectors are widened to
integer values with twice the number of bits as the original
elements.}];
  let Instruction = "vwsub.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = wide_int(a[element]) - wide_int(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = wide_int(a[element]) - wide_int(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vwsub_w>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise widening integer subtraction (second operand)";
  let Description = [{Use these builtins to do an elementwise subtraction of two
integer vectors.

Before doing the subtraction, the elements of the second vector operand are
widened to integer values with twice the number of bits as the original
elements.}];
  let Instruction = "vwsub.wv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] - wide_int(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] + wide_int(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vwsubu>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise widening unsigned integer subtraction";
  let Description = [{Use these builtins to do an elementwise subtraction of two
integer vectors.

Before doing the subtraction, the elements of the two vectors are widened to
integer values with twice the number of bits as the original
elements.}];
  let Instruction = "vwsubu.vv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = wide_int(a[element]) - wide_int(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = wide_int(a[element]) - wide_int(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}

def : DocEPIBuiltin<vwsubu_w>
{
  let Category = DocEPIIntegerArithmetic;
  let Title = "Elementwise widening unsigned integer subtraction (second operand)";
  let Description = [{Use these builtins to do an elementwise subtraction of two
integer vectors.

Before doing the subtraction, the elements of the second vector operand are
widened to integer values with twice the number of bits as the original
elements.}];
  let Instruction = "vwsubu.wv";

  let Operation = [{
    for element = 0 to gvl - 1
       result[element] = a[element] - wide_uint(b[element])

    result[gvl : VLMAX] = 0
  }];

  let OperationMask = [{
    for element = 0 to gvl - 1
       if mask[element] then
         result[element] = a[element] - wide_uint(b[element])
       else
         result[element] = merge[element]

    result[gvl : VLMAX] = 0
  }];
}
