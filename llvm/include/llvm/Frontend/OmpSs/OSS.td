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
def OSSC_Shmem : Clause<"shmem"> {
  let clangClass = "OSSShmemClause";
  let flangClass = "ScalarIntExpr";
}
def OSSC_Onready : Clause<"onready"> {
  let clangClass = "OSSOnreadyClause";
  let flangClass = "Expr";
}
def OSSC_Default : Clause<"default"> {
  let clangClass = "OSSDefaultClause";
  let flangClass = "OSSDefaultClause";
}
def OSSC_Device : Clause<"device"> {
  let clangClass = "OSSDeviceClause";
  let flangClass = "OSSDeviceClause";
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
def OSSC_In : Clause<"in"> {
  let flangClass = "OSSObjectList";
}
def OSSC_Out : Clause<"out"> {
  let flangClass = "OSSObjectList";
}
def OSSC_Inout : Clause<"inout"> {
  let flangClass = "OSSObjectList";
}
def OSSC_On : Clause<"on"> {
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
  let flangClass = "ScalarIntExpr";
}
def OSSC_Collapse : Clause<"collapse"> {
  let clangClass = "OSSCollapseClause";
  let flangClass = "ScalarIntConstantExpr";
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
  let association = AS_Block;
  let category = CA_Executable;
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
    VersionedClause<OSSC_Shmem>,
    VersionedClause<OSSC_Onready>,
    VersionedClause<OSSC_Device>,
    VersionedClause<OSSC_Ndrange>,
  ];
  let association = AS_Declaration;
  let category = CA_Declarative;
}
def OSS_DeclareReduction : Directive<"declare reduction"> {
  let association = AS_None;
  let category = CA_Declarative;
}
def OSS_TaskWait : Directive<"taskwait"> {
  let allowedClauses = [
    VersionedClause<OSSC_Depend>,
    VersionedClause<OSSC_On>,
    VersionedClause<OSSC_In>,
    VersionedClause<OSSC_Out>,
    VersionedClause<OSSC_Inout>,
  ];
  let association = AS_None;
  let category = CA_Executable;
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
  let association = AS_None;
  let category = CA_Executable;
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
  let association = AS_Loop;
  let category = CA_Executable;
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
  let association = AS_Loop;
  let category = CA_Executable;
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
  let association = AS_Loop;
  let category = CA_Executable;
}

def OSS_TaskIterWhile : Directive<"taskiter while"> {
  let association = AS_None;
  let category = CA_Executable;
}
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
  let association = AS_Loop;
  let category = CA_Executable;
}
def OSS_Critical : Directive<"critical"> {
  let association = AS_Block;
  let category = CA_Executable;
}
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
  let association = AS_Block;
  let category = CA_Executable;
}
def OSS_Assert : Directive<"assert"> {
  let association = AS_None;
  let category = CA_Informational;
}
def OSS_Unknown : Directive<"unknown"> {
  let isDefault = true;
  let association = AS_None;
  let category = CA_Utility;
}
