//===-- OmpSsOps.td - OmpSs dialect operation definitions *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the OmpSs dialect.
//
//===----------------------------------------------------------------------===//


#ifndef OMPSS_OPS
#define OMPSS_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/OmpSs/OSSCommon.td"
include "mlir/Dialect/OmpSs/OmpSsOpsInterfaces.td"

def OmpSs_Dialect : Dialect {
  let name = "oss";
  let cppNamespace = "::mlir::oss";

  let useDefaultAttributePrinterParser = 1;
}

// Type which can be constraint accepting standard integers, indices and
// LLVM integer types.
def IntLikeType : AnyTypeOf<[AnyInteger, Index]>;

// Possible values for the default clause
def ClauseDefaultPrivate : I32EnumAttrCase<"defprivate", 0>;
def ClauseDefaultFirstPrivate : I32EnumAttrCase<"deffirstprivate", 1>;
def ClauseDefaultShared : I32EnumAttrCase<"defshared", 2>;
def ClauseDefaultNone : I32EnumAttrCase<"defnone", 3>;

def ClauseDefault : I32EnumAttr<
    "ClauseDefault",
    "default clause",
    [ClauseDefaultPrivate, ClauseDefaultFirstPrivate, ClauseDefaultShared,
     ClauseDefaultNone]> {
  let cppNamespace = "::mlir::oss";
  let genSpecializedAttr = 0;
}
def ClauseDefaultAttr : EnumAttr<OmpSs_Dialect, ClauseDefault,
                                       "default_clause">;

class OmpSs_Op<string mnemonic, list<Trait> traits = []> :
      Op<OmpSs_Dialect, mnemonic, traits> {
  code extraClassDeclarationBase = [{
    static constexpr bool allowsIf() { return true; }
    static constexpr bool allowsFinal() { return true; }
    static constexpr bool allowsCost() { return true; }
    static constexpr bool allowsPriority() { return true; }
    static constexpr bool allowsDefault() { return true; }
    static constexpr bool allowsPrivate() { return true; }
    static constexpr bool allowsFirstprivate() { return true; }
    static constexpr bool allowsShared() { return true; }
    static constexpr bool allowsVlaDims() { return true; }
    static constexpr bool allowsCaptures() { return true; }
    static constexpr bool allowsIn() { return true; }
    static constexpr bool allowsOut() { return true; }
    static constexpr bool allowsInout() { return true; }
    static constexpr bool allowsConcurrent() { return true; }
    static constexpr bool allowsCommutative() { return true; }
    static constexpr bool allowsWeakIn() { return true; }
    static constexpr bool allowsWeakOut() { return true; }
    static constexpr bool allowsWeakInout() { return true; }
    static constexpr bool allowsWeakConcurrent() { return true; }
    static constexpr bool allowsWeakCommutative() { return true; }
  }];
}

// Base class for loop directives like taskloop
class LoopOp<string mnemonic, dag extraArguments,
    string extraArgumentsAssembly> :
      OmpSs_Op<mnemonic,
        [AttrSizedOperandSegments, NoFreeVariables,
          DeclareOpInterfaceMethods<OutlineableOmpSsOpInterface>]> {
  dag baseArguments = (ins IntLikeType:$lower_bound,
    IntLikeType:$upper_bound,
    IntLikeType:$step,
    IntLikeType:$loop_type,
    AnyType:$ind_var,
    Optional<AnyType>:$if_expr_var,
    Optional<AnyType>:$final_expr_var,
    Optional<AnyType>:$cost_expr_var,
    Optional<AnyType>:$priority_expr_var,
    OptionalAttr<ClauseDefaultAttr>:$default_val,
    Variadic<AnyType>:$private_vars,
    Variadic<AnyType>:$private_type_vars,
    Variadic<AnyType>:$firstprivate_vars,
    Variadic<AnyType>:$firstprivate_type_vars,
    Variadic<AnyType>:$copy_vars,
    Variadic<AnyType>:$init_vars,
    Variadic<AnyType>:$deinit_vars,
    Variadic<AnyType>:$shared_vars,
    Variadic<AnyType>:$shared_type_vars,
    Variadic<AnyType>:$vlaDims_vars,
    Variadic<AnyType>:$captures_vars,
    Variadic<AnyType>:$in_vars,
    Variadic<AnyType>:$out_vars,
    Variadic<AnyType>:$inout_vars,
    Variadic<AnyType>:$concurrent_vars,
    Variadic<AnyType>:$commutative_vars,
    Variadic<AnyType>:$weakin_vars,
    Variadic<AnyType>:$weakout_vars,
    Variadic<AnyType>:$weakinout_vars,
    Variadic<AnyType>:$weakconcurrent_vars,
    Variadic<AnyType>:$weakcommutative_vars);

  let arguments = !con(baseArguments, extraArguments);

  let regions = (region AnyRegion:$region);

  string baseArgumentsFormat = [{
    `lower_bound` `(` $lower_bound `:` type($lower_bound) `)`
    `upper_bound` `(` $upper_bound `:` type($upper_bound) `)`
    `step` `(` $step `:` type($step) `)`
    `loop_type` `(` $loop_type `:` type($loop_type) `)`
    `ind_var` `(` $ind_var `:` type($ind_var) `)`
    ( `if` `(` $if_expr_var^ `:` type($if_expr_var) `)` )?
    ( `final` `(` $final_expr_var^ `:` type($final_expr_var) `)` )?
    ( `cost` `(` $cost_expr_var^ `:` type($cost_expr_var) `)` )?
    ( `priority` `(` $priority_expr_var^ `:` type($priority_expr_var) `)` )?
    ( `private` `(` $private_vars^ `:` type($private_vars) `)` )?
    ( `private_type` `(` $private_type_vars^ `:` type($private_type_vars) `)` )?
    ( `firstprivate` `(` $firstprivate_vars^ `:` type($firstprivate_vars) `)` )?
    ( `firstprivate_type` `(` $firstprivate_type_vars^ `:` type($firstprivate_type_vars) `)` )?
    ( `copy` `(` $copy_vars^ `:` type($copy_vars) `)` )?
    ( `init` `(` $init_vars^ `:` type($init_vars) `)` )?
    ( `deinit` `(` $deinit_vars^ `:` type($deinit_vars) `)` )?
    ( `shared` `(` $shared_vars^ `:` type($shared_vars) `)` )?
    ( `shared_type` `(` $shared_type_vars^ `:` type($shared_type_vars) `)` )?
    ( `vlaDims` `(` $vlaDims_vars^ `:` type($vlaDims_vars) `)` )?
    ( `captures` `(` $captures_vars^ `:` type($captures_vars) `)` )?
    ( `in` `(` $in_vars^ `:` type($in_vars) `)` )?
    ( `out` `(` $out_vars^ `:` type($out_vars) `)` )?
    ( `inout` `(` $inout_vars^ `:` type($inout_vars) `)` )?
    ( `concurrent` `(` $concurrent_vars^ `:` type($concurrent_vars) `)` )?
    ( `commutative` `(` $commutative_vars^ `:` type($commutative_vars) `)` )?
    ( `weakin` `(` $weakin_vars^ `:` type($weakin_vars) `)` )?
    ( `weakout` `(` $weakout_vars^ `:` type($weakout_vars) `)` )?
    ( `weakinout` `(` $weakinout_vars^ `:` type($weakinout_vars) `)` )?
    ( `weakconcurrent` `(` $weakconcurrent_vars^ `:` type($weakconcurrent_vars) `)` )?
    ( `weakcommutative` `(` $weakcommutative_vars^ `:` type($weakcommutative_vars) `)` )?
    ( `default` `(` $default_val^ `)` )?
    attr-dict-with-keyword
  }];

  let assemblyFormat = !strconcat(!strconcat(baseArgumentsFormat, extraArgumentsAssembly), "$region");

  let extraClassDeclaration = extraClassDeclarationBase;
}

def DepOp : OmpSs_Op<"dependency", []> {
  let summary = "dependency";
  let description = [{
  }];

  let arguments = (ins AnyType:$base,
                   SymbolNameAttr:$function,
                   Variadic<AnyType>:$arguments);

  let results = (outs IntLikeType:$result);

  let assemblyFormat = [{
    `base` `(` $base `:` type($base) `)`
    `function` `(` $function `)`
    ( `arguments` `(` $arguments^ `:` type($arguments) `)` )?
    `->` type($result)
    attr-dict-with-keyword
  }];
}

def CopyOp : OmpSs_Op<"copy", []> {
  let summary = "copy";
  let description = [{
  }];

  let arguments = (ins AnyType:$base,
                   SymbolNameAttr:$function);

  let results = (outs IntLikeType:$result);

  let assemblyFormat = [{
    `base` `(` $base `:` type($base) `)`
    `function` `(` $function `)`
    `->` type($result)
    attr-dict-with-keyword
  }];
}

def VlaDimOp : OmpSs_Op<"vlaDim", [AttrSizedOperandSegments]> {
  let summary = "vlaDim";
  let description = [{
  }];

  let arguments = (ins AnyType:$pointer,
                   Variadic<AnyType>:$sizes,
                   Variadic<AnyType>:$lbs);

  let results = (outs IntLikeType:$result);

  let assemblyFormat = [{
    `pointer` `(` $pointer `:` type($pointer) `)`
    ( `sizes` `(` $sizes^ `:` type($sizes) `)` )?
    ( `lbs` `(` $lbs^ `:` type($lbs) `)` )?
    `->` type($result)
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// task Construct
//===----------------------------------------------------------------------===//

def TaskOp : OmpSs_Op<"task", [AttrSizedOperandSegments, NoFreeVariables,
          DeclareOpInterfaceMethods<OutlineableOmpSsOpInterface>]> {
  let summary = "task construct";
  let description = [{
  }];

  let arguments = (ins Optional<AnyType>:$if_expr_var,
    Optional<AnyType>:$final_expr_var,
    Optional<AnyType>:$cost_expr_var,
    Optional<AnyType>:$priority_expr_var,
    OptionalAttr<ClauseDefaultAttr>:$default_val,
    Variadic<AnyType>:$private_vars,
    Variadic<AnyType>:$private_type_vars,
    Variadic<AnyType>:$firstprivate_vars,
    Variadic<AnyType>:$firstprivate_type_vars,
    Variadic<AnyType>:$copy_vars,
    Variadic<AnyType>:$init_vars,
    Variadic<AnyType>:$deinit_vars,
    Variadic<AnyType>:$shared_vars,
    Variadic<AnyType>:$shared_type_vars,
    Variadic<AnyType>:$vlaDims_vars,
    Variadic<AnyType>:$captures_vars,
    Variadic<AnyType>:$in_vars,
    Variadic<AnyType>:$out_vars,
    Variadic<AnyType>:$inout_vars,
    Variadic<AnyType>:$concurrent_vars,
    Variadic<AnyType>:$commutative_vars,
    Variadic<AnyType>:$weakin_vars,
    Variadic<AnyType>:$weakout_vars,
    Variadic<AnyType>:$weakinout_vars,
    Variadic<AnyType>:$weakconcurrent_vars,
    Variadic<AnyType>:$weakcommutative_vars);

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    ( `if` `(` $if_expr_var^ `:` type($if_expr_var) `)` )?
    ( `final` `(` $final_expr_var^ `:` type($final_expr_var) `)` )?
    ( `cost` `(` $cost_expr_var^ `:` type($cost_expr_var) `)` )?
    ( `priority` `(` $priority_expr_var^ `:` type($priority_expr_var) `)` )?
    ( `private` `(` $private_vars^ `:` type($private_vars) `)` )?
    ( `private_type` `(` $private_type_vars^ `:` type($private_type_vars) `)` )?
    ( `firstprivate` `(` $firstprivate_vars^ `:` type($firstprivate_vars) `)` )?
    ( `firstprivate_type` `(` $firstprivate_type_vars^ `:` type($firstprivate_type_vars) `)` )?
    ( `copy` `(` $copy_vars^ `:` type($copy_vars) `)` )?
    ( `init` `(` $init_vars^ `:` type($init_vars) `)` )?
    ( `deinit` `(` $deinit_vars^ `:` type($deinit_vars) `)` )?
    ( `shared` `(` $shared_vars^ `:` type($shared_vars) `)` )?
    ( `shared_type` `(` $shared_type_vars^ `:` type($shared_type_vars) `)` )?
    ( `vlaDims` `(` $vlaDims_vars^ `:` type($vlaDims_vars) `)` )?
    ( `captures` `(` $captures_vars^ `:` type($captures_vars) `)` )?
    ( `in` `(` $in_vars^ `:` type($in_vars) `)` )?
    ( `out` `(` $out_vars^ `:` type($out_vars) `)` )?
    ( `inout` `(` $inout_vars^ `:` type($inout_vars) `)` )?
    ( `concurrent` `(` $concurrent_vars^ `:` type($concurrent_vars) `)` )?
    ( `commutative` `(` $commutative_vars^ `:` type($commutative_vars) `)` )?
    ( `weakin` `(` $weakin_vars^ `:` type($weakin_vars) `)` )?
    ( `weakout` `(` $weakout_vars^ `:` type($weakout_vars) `)` )?
    ( `weakinout` `(` $weakinout_vars^ `:` type($weakinout_vars) `)` )?
    ( `weakconcurrent` `(` $weakconcurrent_vars^ `:` type($weakconcurrent_vars) `)` )?
    ( `weakcommutative` `(` $weakcommutative_vars^ `:` type($weakcommutative_vars) `)` )?
    ( `default` `(` $default_val^ `)` )?
    attr-dict-with-keyword
    $region
  }];

  string taskExtraClassDeclaration = [{
    static constexpr bool allowsChunksize() { return false; }
    static constexpr bool allowsGrainsize() { return false; }
  }];
  let extraClassDeclaration = !strconcat(extraClassDeclarationBase, taskExtraClassDeclaration);
}

//===----------------------------------------------------------------------===//
// taskloop Construct
//===----------------------------------------------------------------------===//

def TaskloopOp : LoopOp<"taskloop",
      (ins Optional<AnyType>:$grainsize_expr_var),
      [{
        ( `grainsize` `(` $grainsize_expr_var^ `:` type($grainsize_expr_var) `)` )?
      }]> {
  let summary = "taskloop construct";
  let description = [{
  }];

  string taskloopExtraClassDeclaration = [{
    static constexpr bool allowsChunksize() { return false; }
    static constexpr bool allowsGrainsize() { return true; }
  }];
  let extraClassDeclaration = !strconcat(extraClassDeclarationBase, taskloopExtraClassDeclaration);
}

//===----------------------------------------------------------------------===//
// task for Construct
//===----------------------------------------------------------------------===//

def TaskForOp : LoopOp<"task_for",
      (ins Optional<AnyType>:$chunksize_expr_var),
      [{
        ( `chunksize` `(` $chunksize_expr_var^ `:` type($chunksize_expr_var) `)` )?
      }]> {
  let summary = "task for construct";
  let description = [{
  }];
  string taskForExtraClassDeclaration = [{
    static constexpr bool allowsChunksize() { return true; }
    static constexpr bool allowsGrainsize() { return false; }
  }];
  let extraClassDeclaration = !strconcat(extraClassDeclarationBase, taskForExtraClassDeclaration);
}

//===----------------------------------------------------------------------===//
// taskloop for Construct
//===----------------------------------------------------------------------===//

def TaskloopForOp : LoopOp<"taskloop_for",
      (ins Optional<AnyType>:$chunksize_expr_var,
           Optional<AnyType>:$grainsize_expr_var),
      [{
        ( `chunksize` `(` $chunksize_expr_var^ `:` type($chunksize_expr_var) `)` )?
        ( `grainsize` `(` $grainsize_expr_var^ `:` type($grainsize_expr_var) `)` )?
      }]> {
  let summary = "taskloop for construct";
  let description = [{
  }];
  string taskloopExtraClassDeclaration = [{
    static constexpr bool allowsChunksize() { return true; }
    static constexpr bool allowsGrainsize() { return true; }
  }];
  let extraClassDeclaration = !strconcat(extraClassDeclarationBase, taskloopExtraClassDeclaration);
}

def TerminatorOp : OmpSs_Op<"terminator", [Terminator]> {
  let summary = "terminator for OmpSs-2 regions";
  let description = [{
    A terminator operation for regions that appear in the body of OmpSs-2
    operation.  These regions are not expected to return any value so the
    terminator takes no operands. The terminator op returns control to the
    enclosing op.
  }];

  let assemblyFormat = "attr-dict";
}


//===----------------------------------------------------------------------===//
// taskwait Construct
//===----------------------------------------------------------------------===//

def TaskwaitOp : OmpSs_Op<"taskwait"> {
  let summary = "taskwait construct";
  let description = [{
    The taskwait construct specifies a wait on the completion of child tasks
    of the current task.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// release Construct
//===----------------------------------------------------------------------===//

def ReleaseOp : OmpSs_Op<"release", [AttrSizedOperandSegments]> {
  let summary = "release construct";
  let description = [{
    The release construct specifies a wait on the completion of child tasks
    of the current task.
  }];

  let arguments = (ins Variadic<AnyType>:$in_vars,
    Variadic<AnyType>:$out_vars,
    Variadic<AnyType>:$inout_vars,
    Variadic<AnyType>:$weakin_vars,
    Variadic<AnyType>:$weakout_vars,
    Variadic<AnyType>:$weakinout_vars);

  let assemblyFormat = [{
    ( `in` `(` $in_vars^ `:` type($in_vars) `)` )?
    ( `out` `(` $out_vars^ `:` type($out_vars) `)` )?
    ( `inout` `(` $inout_vars^ `:` type($inout_vars) `)` )?
    ( `weakin` `(` $weakin_vars^ `:` type($weakin_vars) `)` )?
    ( `weakout` `(` $weakout_vars^ `:` type($weakout_vars) `)` )?
    ( `weakinout` `(` $weakinout_vars^ `:` type($weakinout_vars) `)` )?
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    static constexpr bool allowsIf() { return false; }
    static constexpr bool allowsFinal() { return false; }
    static constexpr bool allowsCost() { return false; }
    static constexpr bool allowsPriority() { return false; }
    static constexpr bool allowsDefault() { return false; }
    static constexpr bool allowsPrivate() { return false; }
    static constexpr bool allowsFirstprivate() { return false; }
    static constexpr bool allowsShared() { return false; }
    static constexpr bool allowsVlaDims() { return false; }
    static constexpr bool allowsCaptures() { return false; }
    static constexpr bool allowsIn() { return true; }
    static constexpr bool allowsOut() { return true; }
    static constexpr bool allowsInout() { return true; }
    static constexpr bool allowsConcurrent() { return false; }
    static constexpr bool allowsCommutative() { return false; }
    static constexpr bool allowsWeakIn() { return true; }
    static constexpr bool allowsWeakOut() { return true; }
    static constexpr bool allowsWeakInout() { return true; }
    static constexpr bool allowsWeakConcurrent() { return false; }
    static constexpr bool allowsWeakCommutative() { return false; }
    static constexpr bool allowsChunksize() { return false; }
    static constexpr bool allowsGrainsize() { return false; }
  }];
}

#endif // OMPSS_OPS
