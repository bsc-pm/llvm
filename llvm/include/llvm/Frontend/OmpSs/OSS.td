//===-- OSS.td - OmpSs directive definition file ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for OmpSs directives and clauses.
//
//===----------------------------------------------------------------------===//

include "llvm/Frontend/Directive/DirectiveBase.td"

//===----------------------------------------------------------------------===//
// Definition of general OmpSs information
//===----------------------------------------------------------------------===//

def OmpSs : DirectiveLanguage {
  let name = "OmpSs";
  let cppNamespace = "oss"; // final namespace will be llvm::oss
  let directivePrefix = "OSSD_";
  let clausePrefix = "OSSC_";
  let makeEnumAvailableInNamespace = true;
  let enableBitmaskEnumInNamespace = true;
  let clauseEnumSetClass = "OSSClauseSet";
  let flangClauseBaseClass = "OSSClause";
}

//===----------------------------------------------------------------------===//
// Definition of OmpSs clauses
//===----------------------------------------------------------------------===//

def OSSC_If : Clause<"if"> {
  let clangClass = "OSSIfClause";
  let flangClass = "ScalarLogicalExpr";
}
def OSSC_Final : Clause<"final"> {
  let clangClass = "OSSFinalClause";
  let flangClass = "ScalarLogicalExpr";
}
def OSSC_Cost : Clause<"cost"> {
  let clangClass = "OSSCostClause";
  let flangClass = "ScalarIntExpr";
}
def OSSC_Priority : Clause<"priority"> {
  let clangClass = "OSSPriorityClause";
  let flangClass = "ScalarIntExpr";
}
def OSSC_Label : Clause<"label"> {
  let clangClass = "OSSLabelClause";
  let flangClass = "ScalarDefaultCharExpr";
}
def OSSC_Wait : Clause<"wait"> {
  let clangClass = "OSSWaitClause";
}
def OSSC_Update : Clause<"update"> {
  let clangClass = "OSSUpdateClause";
}
def OSSC_Onready : Clause<"onready"> {
  let clangClass = "OSSOnreadyClause";
}
def OSSC_Default : Clause<"default"> {
  let clangClass = "OSSDefaultClause";
  let flangClass = "OSSDefaultClause";
}
def OSSC_Device : Clause<"device"> {
  let clangClass = "OSSDeviceClause";
}
def OSSC_Private : Clause<"private"> {
  let clangClass = "OSSPrivateClause";
  let flangClass = "OSSObjectList";
}
def OSSC_FirstPrivate : Clause<"firstprivate"> {
  let clangClass = "OSSFirstprivateClause";
  let flangClass = "OSSObjectList";
}
def OSSC_Shared : Clause<"shared"> {
  let clangClass = "OSSSharedClause";
  let flangClass = "OSSObjectList";
}
def OSSC_Depend : Clause<"depend"> {
  let clangClass = "OSSDependClause";
  let flangClass = "OSSDependClause";
}
def OSSC_On : Clause<"on"> {}
def OSSC_In : Clause<"in"> {
  let flangClass = "OSSObjectList";
}
def OSSC_Out : Clause<"out"> {
  let flangClass = "OSSObjectList";
}
def OSSC_Inout : Clause<"inout"> {
  let flangClass = "OSSObjectList";
}
def OSSC_Concurrent : Clause<"concurrent"> {
  let flangClass = "OSSObjectList";
}
def OSSC_Commutative : Clause<"commutative"> {
  let flangClass = "OSSObjectList";
}
def OSSC_WeakIn : Clause<"weakin"> {
  let flangClass = "OSSObjectList";
}
def OSSC_WeakOut : Clause<"weakout"> {
  let flangClass = "OSSObjectList";
}
def OSSC_WeakInout : Clause<"weakinout"> {
  let flangClass = "OSSObjectList";
}
def OSSC_WeakConcurrent : Clause<"weakconcurrent"> {
  let flangClass = "OSSObjectList";
}
def OSSC_WeakCommutative : Clause<"weakcommutative"> {
  let flangClass = "OSSObjectList";
}
def OSSC_Reduction : Clause<"reduction"> {
  let clangClass = "OSSReductionClause";
  let flangClass = "OSSReductionClause";
}
def OSSC_WeakReduction : Clause<"weakreduction"> {
}
def OSSC_GrainSize : Clause<"grainsize"> {
  let clangClass = "OSSGrainsizeClause";
  let flangClass = "ScalarIntExpr";
}
def OSSC_ChunkSize : Clause<"chunksize"> {
  let clangClass = "OSSChunksizeClause";
  let flangClass = "ScalarIntExpr";
}
def OSSC_Unroll : Clause<"unroll"> {
  let clangClass = "OSSUnrollClause";
}
def OSSC_Collapse : Clause<"collapse"> {
  let clangClass = "OSSCollapseClause";
}
def OSSC_Ndrange : Clause<"ndrange"> {
  let clangClass = "OSSNdrangeClause";
}
def OSSC_Read : Clause<"read"> {
  let clangClass = "OSSReadClause";
}
def OSSC_Write : Clause<"write"> {
  let clangClass = "OSSWriteClause";
}
def OSSC_Capture : Clause<"capture"> {
  let clangClass = "OSSCaptureClause";
}
def OSSC_Compare : Clause<"compare"> {
  let clangClass = "OSSCompareClause";
}
def OSSC_SeqCst : Clause<"seq_cst"> {
  let clangClass = "OSSSeqCstClause";
}
def OSSC_AcqRel : Clause<"acq_rel"> {
  let clangClass = "OSSAcqRelClause";
}
def OSSC_Acquire : Clause<"acquire"> {
  let clangClass = "OSSAcquireClause";
}
def OSSC_Release : Clause<"release"> {
  let clangClass = "OSSReleaseClause";
}
def OSSC_Relaxed : Clause<"relaxed"> {
  let clangClass = "OSSRelaxedClause";
}
def OSSC_Unknown : Clause<"unknown"> {
  let isImplicit = true;
  let isDefault = true;
}
def OSSC_Num_instances: Clause<"num_instances"> {
}
def OSSC_Onto: Clause<"onto"> {
}
def OSSC_Num_repetitions: Clause<"num_repetitions"> {
}
def OSSC_Period: Clause<"period"> {
}
def OSSC_Affinity: Clause<"affinity"> {
}
def OSSC_CopyIn: Clause<"copy_in"> {
}
def OSSC_CopyOut: Clause<"copy_out"> {
}
def OSSC_CopyInOut: Clause<"copy_inout"> {
}
def OSSC_Copy_deps: Clause<"copy_deps"> {
}

//===----------------------------------------------------------------------===//
// Definition of OmpSs directives
//===----------------------------------------------------------------------===//

def OSS_Task : Directive<"task"> {
  let allowedClauses = [
    VersionedClause<OSSC_Private>,
    VersionedClause<OSSC_FirstPrivate>,
    VersionedClause<OSSC_Shared>,
    VersionedClause<OSSC_Depend>,
    VersionedClause<OSSC_In>,
    VersionedClause<OSSC_Out>,
    VersionedClause<OSSC_Inout>,
    VersionedClause<OSSC_Concurrent>,
    VersionedClause<OSSC_Commutative>,
    VersionedClause<OSSC_WeakIn>,
    VersionedClause<OSSC_WeakOut>,
    VersionedClause<OSSC_WeakInout>,
    VersionedClause<OSSC_WeakCommutative>,
    VersionedClause<OSSC_Reduction>,
    VersionedClause<OSSC_WeakReduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OSSC_Default>,
    VersionedClause<OSSC_If>,
    VersionedClause<OSSC_Final>,
    VersionedClause<OSSC_Cost>,
    VersionedClause<OSSC_Priority>,
    VersionedClause<OSSC_Label>,
    VersionedClause<OSSC_Wait>,
    VersionedClause<OSSC_Onready>,
    VersionedClause<OSSC_Device>,
  ];
}
def OSS_DeclareTask : Directive<"declare task"> {
  let allowedClauses = [
    VersionedClause<OSSC_Depend>,
    VersionedClause<OSSC_In>,
    VersionedClause<OSSC_Out>,
    VersionedClause<OSSC_Inout>,
    VersionedClause<OSSC_Concurrent>,
    VersionedClause<OSSC_Commutative>,
    VersionedClause<OSSC_WeakIn>,
    VersionedClause<OSSC_WeakOut>,
    VersionedClause<OSSC_WeakInout>,
    VersionedClause<OSSC_WeakCommutative>,
    VersionedClause<OSSC_Reduction>,
    VersionedClause<OSSC_WeakReduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OSSC_If>,
    VersionedClause<OSSC_Final>,
    VersionedClause<OSSC_Cost>,
    VersionedClause<OSSC_Priority>,
    VersionedClause<OSSC_Label>,
    VersionedClause<OSSC_Wait>,
    VersionedClause<OSSC_Onready>,
    VersionedClause<OSSC_Device>,
    VersionedClause<OSSC_Ndrange>,
    VersionedClause<OSSC_Num_instances>,
    VersionedClause<OSSC_Onto>,
    VersionedClause<OSSC_Num_repetitions>,
    VersionedClause<OSSC_Affinity>,
    VersionedClause<OSSC_Period>,
    VersionedClause<OSSC_CopyIn>,
    VersionedClause<OSSC_CopyOut>,
    VersionedClause<OSSC_CopyInOut>,
    VersionedClause<OSSC_Copy_deps>,
  ];
}
def OSS_DeclareReduction : Directive<"declare reduction"> {}
def OSS_TaskWait : Directive<"taskwait"> {
  let allowedClauses = [
    VersionedClause<OSSC_Depend>,
    VersionedClause<OSSC_On>,
    VersionedClause<OSSC_In>,
    VersionedClause<OSSC_Out>,
    VersionedClause<OSSC_Inout>,
  ];
}

def OSS_Release : Directive<"release"> {
  let allowedClauses = [
    VersionedClause<OSSC_Depend>,
    VersionedClause<OSSC_In>,
    VersionedClause<OSSC_Out>,
    VersionedClause<OSSC_Inout>,
    VersionedClause<OSSC_WeakIn>,
    VersionedClause<OSSC_WeakOut>,
    VersionedClause<OSSC_WeakInout>,
  ];
}
def OSS_TaskLoop : Directive<"taskloop"> {
  let allowedClauses = [
    VersionedClause<OSSC_Private>,
    VersionedClause<OSSC_FirstPrivate>,
    VersionedClause<OSSC_Shared>,
    VersionedClause<OSSC_Depend>,
    VersionedClause<OSSC_In>,
    VersionedClause<OSSC_Out>,
    VersionedClause<OSSC_Inout>,
    VersionedClause<OSSC_Concurrent>,
    VersionedClause<OSSC_Commutative>,
    VersionedClause<OSSC_WeakIn>,
    VersionedClause<OSSC_WeakOut>,
    VersionedClause<OSSC_WeakInout>,
    VersionedClause<OSSC_WeakCommutative>,
    VersionedClause<OSSC_Reduction>,
    VersionedClause<OSSC_WeakReduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OSSC_Default>,
    VersionedClause<OSSC_If>,
    VersionedClause<OSSC_Final>,
    VersionedClause<OSSC_Cost>,
    VersionedClause<OSSC_Priority>,
    VersionedClause<OSSC_Label>,
    VersionedClause<OSSC_Wait>,
    VersionedClause<OSSC_GrainSize>,
    VersionedClause<OSSC_Collapse>,
  ];
}
def OSS_TaskFor : Directive<"task for"> {
  let allowedClauses = [
    VersionedClause<OSSC_Private>,
    VersionedClause<OSSC_FirstPrivate>,
    VersionedClause<OSSC_Shared>,
    VersionedClause<OSSC_Depend>,
    VersionedClause<OSSC_In>,
    VersionedClause<OSSC_Out>,
    VersionedClause<OSSC_Inout>,
    VersionedClause<OSSC_Concurrent>,
    VersionedClause<OSSC_Commutative>,
    VersionedClause<OSSC_WeakIn>,
    VersionedClause<OSSC_WeakOut>,
    VersionedClause<OSSC_WeakInout>,
    VersionedClause<OSSC_WeakCommutative>,
    VersionedClause<OSSC_Reduction>,
    VersionedClause<OSSC_WeakReduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OSSC_Default>,
    VersionedClause<OSSC_If>,
    VersionedClause<OSSC_Final>,
    VersionedClause<OSSC_Cost>,
    VersionedClause<OSSC_Priority>,
    VersionedClause<OSSC_Label>,
    VersionedClause<OSSC_Wait>,
    VersionedClause<OSSC_ChunkSize>,
    VersionedClause<OSSC_Collapse>,
  ];
}
def OSS_TaskLoopFor : Directive<"taskloop for"> {
  let allowedClauses = [
    VersionedClause<OSSC_Private>,
    VersionedClause<OSSC_FirstPrivate>,
    VersionedClause<OSSC_Shared>,
    VersionedClause<OSSC_Depend>,
    VersionedClause<OSSC_In>,
    VersionedClause<OSSC_Out>,
    VersionedClause<OSSC_Inout>,
    VersionedClause<OSSC_Concurrent>,
    VersionedClause<OSSC_Commutative>,
    VersionedClause<OSSC_WeakIn>,
    VersionedClause<OSSC_WeakOut>,
    VersionedClause<OSSC_WeakInout>,
    VersionedClause<OSSC_WeakCommutative>,
    VersionedClause<OSSC_Reduction>,
    VersionedClause<OSSC_WeakReduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OSSC_Default>,
    VersionedClause<OSSC_If>,
    VersionedClause<OSSC_Final>,
    VersionedClause<OSSC_Cost>,
    VersionedClause<OSSC_Priority>,
    VersionedClause<OSSC_Label>,
    VersionedClause<OSSC_Wait>,
    VersionedClause<OSSC_ChunkSize>,
    VersionedClause<OSSC_GrainSize>,
    VersionedClause<OSSC_Collapse>,
  ];
}

def OSS_TaskIterWhile : Directive<"taskiter while"> {}
def OSS_TaskIter : Directive<"taskiter"> {
  let allowedClauses = [
    VersionedClause<OSSC_Private>,
    VersionedClause<OSSC_FirstPrivate>,
    VersionedClause<OSSC_Shared>,
    VersionedClause<OSSC_Depend>,
    VersionedClause<OSSC_In>,
    VersionedClause<OSSC_Out>,
    VersionedClause<OSSC_Inout>,
    VersionedClause<OSSC_Concurrent>,
    VersionedClause<OSSC_Commutative>,
    VersionedClause<OSSC_WeakIn>,
    VersionedClause<OSSC_WeakOut>,
    VersionedClause<OSSC_WeakInout>,
    VersionedClause<OSSC_WeakCommutative>,
    VersionedClause<OSSC_Reduction>,
    VersionedClause<OSSC_WeakReduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OSSC_Default>,
    VersionedClause<OSSC_If>,
    VersionedClause<OSSC_Final>,
    VersionedClause<OSSC_Cost>,
    VersionedClause<OSSC_Priority>,
    VersionedClause<OSSC_Label>,
    VersionedClause<OSSC_Wait>,
    VersionedClause<OSSC_Onready>,
    VersionedClause<OSSC_Update>,
    VersionedClause<OSSC_Unroll>,
  ];
}
def OSS_Critical : Directive<"critical"> {}
def OSS_Atomic : Directive<"atomic"> {
  let allowedClauses = [
    VersionedClause<OSSC_Read>,
    VersionedClause<OSSC_Write>,
    VersionedClause<OSSC_Update>,
    VersionedClause<OSSC_Capture>,
    // VersionedClause<OSSC_Compare>
  ];
  let allowedOnceClauses = [
    VersionedClause<OSSC_SeqCst>,
    VersionedClause<OSSC_AcqRel>,
    VersionedClause<OSSC_Acquire>,
    VersionedClause<OSSC_Release>,
    VersionedClause<OSSC_Relaxed>,
  ];
}
def OSS_Assert : Directive<"assert"> {}
def OSS_Unknown : Directive<"unknown"> {
  let isDefault = true;
}
