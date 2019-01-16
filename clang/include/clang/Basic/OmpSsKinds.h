//===--- OmpSsKinds.h - OmpSs enums ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines some OmpSs-specific enums and functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OMPSSKINDS_H
#define LLVM_CLANG_BASIC_OMPSSKINDS_H

#include "llvm/ADT/StringRef.h"

namespace clang {

/// OmpSs directives.
enum OmpSsDirectiveKind {
#define OMPSS_DIRECTIVE(Name) \
  OSSD_##Name,
#define OMPSS_DIRECTIVE_EXT(Name, Str) \
  OSSD_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSD_unknown
};

/// OmpSs clauses.
enum OmpSsClauseKind {
#define OMPSS_CLAUSE(Name, Class) \
  OSSC_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_threadprivate,
  OSSC_uniform,
  OSSC_unknown
};

/// OmpSs attributes for 'default' clause.
enum OmpSsDefaultClauseKind {
#define OMPSS_DEFAULT_KIND(Name) \
  OSSC_DEFAULT_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_DEFAULT_unknown
};

/// OmpSs attributes for 'proc_bind' clause.
enum OmpSsProcBindClauseKind {
#define OMPSS_PROC_BIND_KIND(Name) \
  OSSC_PROC_BIND_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_PROC_BIND_unknown
};

/// OmpSs attributes for 'schedule' clause.
enum OmpSsScheduleClauseKind {
#define OMPSS_SCHEDULE_KIND(Name) \
  OSSC_SCHEDULE_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_SCHEDULE_unknown
};

/// OmpSs modifiers for 'schedule' clause.
enum OmpSsScheduleClauseModifier {
  OSSC_SCHEDULE_MODIFIER_unknown = OSSC_SCHEDULE_unknown,
#define OMPSS_SCHEDULE_MODIFIER(Name) \
  OSSC_SCHEDULE_MODIFIER_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_SCHEDULE_MODIFIER_last
};

/// OmpSs attributes for 'depend' clause.
enum OmpSsDependClauseKind {
#define OMPSS_DEPEND_KIND(Name) \
  OSSC_DEPEND_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_DEPEND_unknown
};

/// OmpSs attributes for 'linear' clause.
enum OmpSsLinearClauseKind {
#define OMPSS_LINEAR_KIND(Name) \
  OSSC_LINEAR_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_LINEAR_unknown
};

/// OmpSs mapping kind for 'map' clause.
enum OmpSsMapClauseKind {
#define OMPSS_MAP_KIND(Name) \
  OSSC_MAP_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_MAP_unknown
};

/// OmpSs attributes for 'dist_schedule' clause.
enum OmpSsDistScheduleClauseKind {
#define OMPSS_DIST_SCHEDULE_KIND(Name) OSSC_DIST_SCHEDULE_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_DIST_SCHEDULE_unknown
};

/// OmpSs attributes for 'defaultmap' clause.
enum OmpSsDefaultmapClauseKind {
#define OMPSS_DEFAULTMAP_KIND(Name) \
  OSSC_DEFAULTMAP_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_DEFAULTMAP_unknown
};

/// OmpSs modifiers for 'defaultmap' clause.
enum OmpSsDefaultmapClauseModifier {
  OSSC_DEFAULTMAP_MODIFIER_unknown = OSSC_DEFAULTMAP_unknown,
#define OMPSS_DEFAULTMAP_MODIFIER(Name) \
  OSSC_DEFAULTMAP_MODIFIER_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_DEFAULTMAP_MODIFIER_last
};

/// Scheduling data for loop-based OmpSs directives.
struct OmpSsScheduleTy final {
  OmpSsScheduleClauseKind Schedule = OSSC_SCHEDULE_unknown;
  OmpSsScheduleClauseModifier M1 = OSSC_SCHEDULE_MODIFIER_unknown;
  OmpSsScheduleClauseModifier M2 = OSSC_SCHEDULE_MODIFIER_unknown;
};

OmpSsDirectiveKind getOmpSsDirectiveKind(llvm::StringRef Str);
const char *getOmpSsDirectiveName(OmpSsDirectiveKind Kind);

OmpSsClauseKind getOmpSsClauseKind(llvm::StringRef Str);
const char *getOmpSsClauseName(OmpSsClauseKind Kind);

unsigned getOmpSsSimpleClauseType(OmpSsClauseKind Kind, llvm::StringRef Str);
const char *getOmpSsSimpleClauseTypeName(OmpSsClauseKind Kind, unsigned Type);

bool isAllowedClauseForDirective(OmpSsDirectiveKind DKind,
                                 OmpSsClauseKind CKind);

/// Checks if the specified directive is a directive with an associated
/// loop construct.
/// \param DKind Specified directive.
/// \return true - the directive is a loop-associated directive like 'omp simd'
/// or 'omp for' directive, otherwise - false.
bool isOmpSsLoopDirective(OmpSsDirectiveKind DKind);

/// Checks if the specified directive is a worksharing directive.
/// \param DKind Specified directive.
/// \return true - the directive is a worksharing directive like 'omp for',
/// otherwise - false.
bool isOmpSsWorksharingDirective(OmpSsDirectiveKind DKind);

/// Checks if the specified directive is a taskloop directive.
/// \param DKind Specified directive.
/// \return true - the directive is a worksharing directive like 'omp taskloop',
/// otherwise - false.
bool isOmpSsTaskLoopDirective(OmpSsDirectiveKind DKind);

/// Checks if the specified directive is a parallel-kind directive.
/// \param DKind Specified directive.
/// \return true - the directive is a parallel-like directive like 'omp
/// parallel', otherwise - false.
bool isOmpSsParallelDirective(OmpSsDirectiveKind DKind);

/// Checks if the specified directive is a target code offload directive.
/// \param DKind Specified directive.
/// \return true - the directive is a target code offload directive like
/// 'omp target', 'omp target parallel', 'omp target xxx'
/// otherwise - false.
bool isOmpSsTargetExecutionDirective(OmpSsDirectiveKind DKind);

/// Checks if the specified directive is a target data offload directive.
/// \param DKind Specified directive.
/// \return true - the directive is a target data offload directive like
/// 'omp target data', 'omp target update', 'omp target enter data',
/// 'omp target exit data'
/// otherwise - false.
bool isOmpSsTargetDataManagementDirective(OmpSsDirectiveKind DKind);

/// Checks if the specified composite/combined directive constitutes a teams
/// directive in the outermost nest.  For example
/// 'omp teams distribute' or 'omp teams distribute parallel for'.
/// \param DKind Specified directive.
/// \return true - the directive has teams on the outermost nest, otherwise -
/// false.
bool isOmpSsNestingTeamsDirective(OmpSsDirectiveKind DKind);

/// Checks if the specified directive is a teams-kind directive.  For example,
/// 'omp teams distribute' or 'omp target teams'.
/// \param DKind Specified directive.
/// \return true - the directive is a teams-like directive, otherwise - false.
bool isOmpSsTeamsDirective(OmpSsDirectiveKind DKind);

/// Checks if the specified directive is a simd directive.
/// \param DKind Specified directive.
/// \return true - the directive is a simd directive like 'omp simd',
/// otherwise - false.
bool isOmpSsSimdDirective(OmpSsDirectiveKind DKind);

/// Checks if the specified directive is a distribute directive.
/// \param DKind Specified directive.
/// \return true - the directive is a distribute-directive like 'omp
/// distribute',
/// otherwise - false.
bool isOmpSsDistributeDirective(OmpSsDirectiveKind DKind);

/// Checks if the specified composite/combined directive constitutes a
/// distribute directive in the outermost nest.  For example,
/// 'omp distribute parallel for' or 'omp distribute'.
/// \param DKind Specified directive.
/// \return true - the directive has distribute on the outermost nest.
/// otherwise - false.
bool isOmpSsNestingDistributeDirective(OmpSsDirectiveKind DKind);

/// Checks if the specified clause is one of private clauses like
/// 'private', 'firstprivate', 'reduction' etc..
/// \param Kind Clause kind.
/// \return true - the clause is a private clause, otherwise - false.
bool isOmpSsPrivate(OmpSsClauseKind Kind);

/// Checks if the specified clause is one of threadprivate clauses like
/// 'threadprivate', 'copyin' or 'copyprivate'.
/// \param Kind Clause kind.
/// \return true - the clause is a threadprivate clause, otherwise - false.
bool isOmpSsThreadPrivate(OmpSsClauseKind Kind);

/// Checks if the specified directive kind is one of tasking directives - task,
/// taskloop or taksloop simd.
bool isOmpSsTaskingDirective(OmpSsDirectiveKind Kind);

/// Checks if the specified directive kind is one of the composite or combined
/// directives that need loop bound sharing across loops outlined in nested
/// functions
bool isOmpSsLoopBoundSharingDirective(OmpSsDirectiveKind Kind);

/// Return the captured regions of an OmpSs directive.
void getOmpSsCaptureRegions(
    llvm::SmallVectorImpl<OmpSsDirectiveKind> &CaptureRegions,
    OmpSsDirectiveKind DKind);
}

#endif
