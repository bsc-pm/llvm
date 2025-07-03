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

def OSSC_Immediate : Clause<[Spelling<"immediate">]> {
  let clangClass = "OSSImmediateClause";
  let flangClass = "ScalarLogicalExpr";
}
def OSSC_Microtask : Clause<[Spelling<"microtask">]> {
  let clangClass = "OSSMicrotaskClause";
  let flangClass = "ScalarLogicalExpr";
}
def OSSC_If : Clause<[Spelling<"if">]> {
  let clangClass = "OSSIfClause";
  let flangClass = "ScalarLogicalExpr";
}
def OSSC_Final : Clause<[Spelling<"final">]> {
  let clangClass = "OSSFinalClause";
  let flangClass = "ScalarLogicalExpr";
}
def OSSC_Cost : Clause<[Spelling<"cost">]> {
  let clangClass = "OSSCostClause";
  let flangClass = "ScalarIntExpr";
}
def OSSC_Priority : Clause<[Spelling<"priority">]> {
  let clangClass = "OSSPriorityClause";
  let flangClass = "ScalarIntExpr";
}
def OSSC_Label : Clause<[Spelling<"label">]> {
  let clangClass = "OSSLabelClause";
  let flangClass = "ScalarDefaultCharExpr";
}
def OSSC_Wait : Clause<[Spelling<"wait">]> {
  let clangClass = "OSSWaitClause";
}
def OSSC_Update : Clause<[Spelling<"update">]> {
  let clangClass = "OSSUpdateClause";
}
def OSSC_Shmem : Clause<[Spelling<"shmem">]> {
  let clangClass = "OSSShmemClause";
  let flangClass = "ScalarIntExpr";
}
def OSSC_Onready : Clause<[Spelling<"onready">]> {
  let clangClass = "OSSOnreadyClause";
  let flangClass = "Expr";
}
def OSSC_Default : Clause<[Spelling<"default">]> {
  let clangClass = "OSSDefaultClause";
  let flangClass = "OSSDefaultClause";
}
def OSSC_Device : Clause<[Spelling<"device">]> {
  let clangClass = "OSSDeviceClause";
  let flangClass = "OSSDeviceClause";
}
def OSSC_Private : Clause<[Spelling<"private">]> {
  let clangClass = "OSSPrivateClause";
  let flangClass = "OSSObjectList";
}
def OSSC_FirstPrivate : Clause<[Spelling<"firstprivate">]> {
  let clangClass = "OSSFirstprivateClause";
  let flangClass = "OSSObjectList";
}
def OSSC_Shared : Clause<[Spelling<"shared">]> {
  let clangClass = "OSSSharedClause";
  let flangClass = "OSSObjectList";
}
def OSSC_Depend : Clause<[Spelling<"depend">]> {
  let clangClass = "OSSDependClause";
  let flangClass = "OSSDependClause";
}
def OSSC_In : Clause<[Spelling<"in">]> {
  let flangClass = "OSSObjectList";
}
def OSSC_Out : Clause<[Spelling<"out">]> {
  let flangClass = "OSSObjectList";
}
def OSSC_Inout : Clause<[Spelling<"inout">]> {
  let flangClass = "OSSObjectList";
}
def OSSC_On : Clause<[Spelling<"on">]> {
  let flangClass = "OSSObjectList";
}
def OSSC_Concurrent : Clause<[Spelling<"concurrent">]> {
  let flangClass = "OSSObjectList";
}
def OSSC_Commutative : Clause<[Spelling<"commutative">]> {
  let flangClass = "OSSObjectList";
}
def OSSC_WeakIn : Clause<[Spelling<"weakin">]> {
  let flangClass = "OSSObjectList";
}
def OSSC_WeakOut : Clause<[Spelling<"weakout">]> {
  let flangClass = "OSSObjectList";
}
def OSSC_WeakInout : Clause<[Spelling<"weakinout">]> {
  let flangClass = "OSSObjectList";
}
def OSSC_WeakConcurrent : Clause<[Spelling<"weakconcurrent">]> {
  let flangClass = "OSSObjectList";
}
def OSSC_WeakCommutative : Clause<[Spelling<"weakcommutative">]> {
  let flangClass = "OSSObjectList";
}
def OSSC_Reduction : Clause<[Spelling<"reduction">]> {
  let clangClass = "OSSReductionClause";
  let flangClass = "OSSReductionClause";
}
def OSSC_WeakReduction : Clause<[Spelling<"weakreduction">]> {
}
def OSSC_GrainSize : Clause<[Spelling<"grainsize">]> {
  let clangClass = "OSSGrainsizeClause";
  let flangClass = "ScalarIntExpr";
}
def OSSC_ChunkSize : Clause<[Spelling<"chunksize">]> {
  let clangClass = "OSSChunksizeClause";
  let flangClass = "ScalarIntExpr";
}
def OSSC_Unroll : Clause<[Spelling<"unroll">]> {
  let clangClass = "OSSUnrollClause";
  let flangClass = "ScalarIntExpr";
}
def OSSC_Collapse : Clause<[Spelling<"collapse">]> {
  let clangClass = "OSSCollapseClause";
  let flangClass = "ScalarIntConstantExpr";
}
def OSSC_Ndrange : Clause<[Spelling<"ndrange">]> {
  let clangClass = "OSSNdrangeClause";
}
def OSSC_Grid : Clause<[Spelling<"grid">]> {
  let clangClass = "OSSGridClause";
}
def OSSC_Read : Clause<[Spelling<"read">]> {
  let clangClass = "OSSReadClause";
}
def OSSC_Write : Clause<[Spelling<"write">]> {
  let clangClass = "OSSWriteClause";
}
def OSSC_Capture : Clause<[Spelling<"capture">]> {
  let clangClass = "OSSCaptureClause";
}
def OSSC_Compare : Clause<[Spelling<"compare">]> {
  let clangClass = "OSSCompareClause";
}
def OSSC_SeqCst : Clause<[Spelling<"seq_cst">]> {
  let clangClass = "OSSSeqCstClause";
}
def OSSC_AcqRel : Clause<[Spelling<"acq_rel">]> {
  let clangClass = "OSSAcqRelClause";
}
def OSSC_Acquire : Clause<[Spelling<"acquire">]> {
  let clangClass = "OSSAcquireClause";
}
def OSSC_Release : Clause<[Spelling<"release">]> {
  let clangClass = "OSSReleaseClause";
}
def OSSC_Relaxed : Clause<[Spelling<"relaxed">]> {
  let clangClass = "OSSRelaxedClause";
}
def OSSC_Unknown : Clause<[Spelling<"unknown">]> {
  let isImplicit = true;
  let isDefault = true;
}

//===----------------------------------------------------------------------===//
// Definition of OmpSs directives
//===----------------------------------------------------------------------===//

def OSS_Task : Directive<[Spelling<"task">]> {
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
def OSS_DeclareTask : Directive<[Spelling<"declare task">]> {
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
    VersionedClause<OSSC_Grid>,
  ];
  let association = AS_Declaration;
  let category = CA_Declarative;
}
def OSS_DeclareReduction : Directive<[Spelling<"declare reduction">]> {
  let association = AS_None;
  let category = CA_Declarative;
}
def OSS_TaskWait : Directive<[Spelling<"taskwait">]> {
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

def OSS_Release : Directive<[Spelling<"release">]> {
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
  ];
  let association = AS_None;
  let category = CA_Executable;
}
def OSS_TaskLoop : Directive<[Spelling<"taskloop">]> {
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
def OSS_TaskFor : Directive<[Spelling<"task for">]> {
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
def OSS_TaskLoopFor : Directive<[Spelling<"taskloop for">]> {
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

def OSS_TaskIterWhile : Directive<[Spelling<"taskiter while">]> {
  let association = AS_None;
  let category = CA_Executable;
}
def OSS_TaskIter : Directive<[Spelling<"taskiter">]> {
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
def OSS_Critical : Directive<[Spelling<"critical">]> {
  let association = AS_Block;
  let category = CA_Executable;
}
def OSS_Atomic : Directive<[Spelling<"atomic">]> {
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
def OSS_Assert : Directive<[Spelling<"assert">]> {
  let association = AS_None;
  let category = CA_Informational;
}
def OSS_Unknown : Directive<[Spelling<"unknown">]> {
  let isDefault = true;
  let association = AS_None;
  let category = CA_Utility;
}
