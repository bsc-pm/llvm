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
  let includeHeader = "llvm/Frontend/OmpSs/OSS.h.inc";
  let clauseEnumSetClass = "OSSClauseSet";
  let flangClauseBaseClass = "OSSClause";
}

//===----------------------------------------------------------------------===//
// Definition of OmpSs clauses
//===----------------------------------------------------------------------===//

def OSSC_If : Clause<"if"> {
  let clangClass = "OSSIfClause";
  let flangClassValue = "ScalarLogicalExpr";
}
def OSSC_Final : Clause<"final"> {
  let clangClass = "OSSFinalClause";
  let flangClassValue = "ScalarLogicalExpr";
}
def OSSC_Cost : Clause<"cost"> {
  let clangClass = "OSSCostClause";
  let flangClassValue = "ScalarIntExpr";
}
def OSSC_Priority : Clause<"priority"> {
  let clangClass = "OSSPriorityClause";
  let flangClassValue = "ScalarIntExpr";
}
def OSSC_Label : Clause<"label"> {
  let clangClass = "OSSLabelClause";
  let flangClassValue = "ScalarDefaultCharExpr";
}
def OSSC_Wait : Clause<"wait"> {
  let clangClass = "OSSWaitClause";
}
def OSSC_Default : Clause<"default"> {
  let clangClass = "OSSDefaultClause";
  let flangClass = "OSSDefaultClause";
}
def OSSC_Private : Clause<"private"> {
  let clangClass = "OSSPrivateClause";
  let flangClassValue = "OSSObjectList";
}
def OSSC_FirstPrivate : Clause<"firstprivate"> {
  let clangClass = "OSSFirstprivateClause";
  let flangClassValue = "OSSObjectList";
}
def OSSC_Shared : Clause<"shared"> {
  let clangClass = "OSSSharedClause";
  let flangClassValue = "OSSObjectList";
}
def OSSC_Depend : Clause<"depend"> {
  let clangClass = "OSSDependClause";
  let flangClass = "OSSDependClause";
}
def OSSC_Reduction : Clause<"reduction"> {
  let clangClass = "OSSReductionClause";
  let flangClass = "OSSReductionClause";
}
def OSSC_GrainSize : Clause<"grainsize"> {
  let clangClass = "OSSGrainsizeClause";
  let flangClassValue = "ScalarIntExpr";
}
def OSSC_ChunkSize : Clause<"chunksize"> {
  let clangClass = "OSSChunksizeClause";
  let flangClassValue = "ScalarIntExpr";
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
    VersionedClause<OSSC_Reduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OSSC_Default>,
    VersionedClause<OSSC_If>,
    VersionedClause<OSSC_Final>,
    VersionedClause<OSSC_Cost>,
    VersionedClause<OSSC_Priority>,
    VersionedClause<OSSC_Label>,
    VersionedClause<OSSC_Wait>,
  ];
}
def OSS_TaskWait : Directive<"taskwait"> {
  let allowedClauses = [
    VersionedClause<OSSC_Depend>,
  ];
}

def OSS_Release : Directive<"release"> {
  let allowedClauses = [
    VersionedClause<OSSC_Depend>,
  ];
}
def OSS_TaskLoop : Directive<"taskloop"> {
  let allowedClauses = [
    VersionedClause<OSSC_Private>,
    VersionedClause<OSSC_FirstPrivate>,
    VersionedClause<OSSC_Shared>,
    VersionedClause<OSSC_Depend>,
    VersionedClause<OSSC_Reduction>,
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
  ];
}
def OSS_TaskFor : Directive<"task for"> {
  let allowedClauses = [
    VersionedClause<OSSC_Private>,
    VersionedClause<OSSC_FirstPrivate>,
    VersionedClause<OSSC_Shared>,
    VersionedClause<OSSC_Depend>,
    VersionedClause<OSSC_Reduction>,
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
  ];
}
def OSS_TaskLoopFor : Directive<"taskloop for"> {
  let allowedClauses = [
    VersionedClause<OSSC_Private>,
    VersionedClause<OSSC_FirstPrivate>,
    VersionedClause<OSSC_Shared>,
    VersionedClause<OSSC_Depend>,
    VersionedClause<OSSC_Reduction>,
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
  ];
}
def OSS_Unknown : Directive<"unknown"> {
  let isDefault = true;
}
