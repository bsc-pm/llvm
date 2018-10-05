//===--- OmpSsKinds.cpp - Token Kinds Support ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the OmpSs enum and support functions.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/OmpSsKinds.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

using namespace clang;

OmpSsDirectiveKind clang::getOmpSsDirectiveKind(StringRef Str) {
  return llvm::StringSwitch<OmpSsDirectiveKind>(Str)
#define OMPSS_DIRECTIVE(Name) .Case(#Name, OSSD_##Name)
#define OMPSS_DIRECTIVE_EXT(Name, Str) .Case(Str, OSSD_##Name)
#include "clang/Basic/OmpSsKinds.def"
      .Default(OSSD_unknown);
}

const char *clang::getOmpSsDirectiveName(OmpSsDirectiveKind Kind) {
  assert(Kind <= OSSD_unknown);
  switch (Kind) {
  case OSSD_unknown:
    return "unknown";
#define OMPSS_DIRECTIVE(Name)                                                 \
  case OSSD_##Name:                                                            \
    return #Name;
#define OMPSS_DIRECTIVE_EXT(Name, Str)                                        \
  case OSSD_##Name:                                                            \
    return Str;
#include "clang/Basic/OmpSsKinds.def"
    break;
  }
  llvm_unreachable("Invalid OmpSs directive kind");
}

OmpSsClauseKind clang::getOmpSsClauseKind(StringRef Str) {
  // 'flush' clause cannot be specified explicitly, because this is an implicit
  // clause for 'flush' directive. If the 'flush' clause is explicitly specified
  // the Parser should generate a warning about extra tokens at the end of the
  // directive.
  if (Str == "flush")
    return OSSC_unknown;
  return llvm::StringSwitch<OmpSsClauseKind>(Str)
#define OMPSS_CLAUSE(Name, Class) .Case(#Name, OSSC_##Name)
#include "clang/Basic/OmpSsKinds.def"
      .Case("uniform", OSSC_uniform)
      .Default(OSSC_unknown);
}

const char *clang::getOmpSsClauseName(OmpSsClauseKind Kind) {
  assert(Kind <= OSSC_unknown);
  switch (Kind) {
  case OSSC_unknown:
    return "unknown";
#define OMPSS_CLAUSE(Name, Class)                                             \
  case OSSC_##Name:                                                            \
    return #Name;
#include "clang/Basic/OmpSsKinds.def"
  case OSSC_uniform:
    return "uniform";
  case OSSC_threadprivate:
    return "threadprivate or thread local";
  }
  llvm_unreachable("Invalid OmpSs clause kind");
}

unsigned clang::getOmpSsSimpleClauseType(OmpSsClauseKind Kind,
                                          StringRef Str) {
  switch (Kind) {
  case OSSC_default:
    return llvm::StringSwitch<OmpSsDefaultClauseKind>(Str)
#define OMPSS_DEFAULT_KIND(Name) .Case(#Name, OSSC_DEFAULT_##Name)
#include "clang/Basic/OmpSsKinds.def"
        .Default(OSSC_DEFAULT_unknown);
  case OSSC_proc_bind:
    return llvm::StringSwitch<OmpSsProcBindClauseKind>(Str)
#define OMPSS_PROC_BIND_KIND(Name) .Case(#Name, OSSC_PROC_BIND_##Name)
#include "clang/Basic/OmpSsKinds.def"
        .Default(OSSC_PROC_BIND_unknown);
  case OSSC_schedule:
    return llvm::StringSwitch<unsigned>(Str)
#define OMPSS_SCHEDULE_KIND(Name)                                             \
  .Case(#Name, static_cast<unsigned>(OSSC_SCHEDULE_##Name))
#define OMPSS_SCHEDULE_MODIFIER(Name)                                         \
  .Case(#Name, static_cast<unsigned>(OSSC_SCHEDULE_MODIFIER_##Name))
#include "clang/Basic/OmpSsKinds.def"
        .Default(OSSC_SCHEDULE_unknown);
  case OSSC_depend:
    return llvm::StringSwitch<OmpSsDependClauseKind>(Str)
#define OMPSS_DEPEND_KIND(Name) .Case(#Name, OSSC_DEPEND_##Name)
#include "clang/Basic/OmpSsKinds.def"
        .Default(OSSC_DEPEND_unknown);
  case OSSC_linear:
    return llvm::StringSwitch<OmpSsLinearClauseKind>(Str)
#define OMPSS_LINEAR_KIND(Name) .Case(#Name, OSSC_LINEAR_##Name)
#include "clang/Basic/OmpSsKinds.def"
        .Default(OSSC_LINEAR_unknown);
  case OSSC_map:
    return llvm::StringSwitch<OmpSsMapClauseKind>(Str)
#define OMPSS_MAP_KIND(Name) .Case(#Name, OSSC_MAP_##Name)
#include "clang/Basic/OmpSsKinds.def"
        .Default(OSSC_MAP_unknown);
  case OSSC_dist_schedule:
    return llvm::StringSwitch<OmpSsDistScheduleClauseKind>(Str)
#define OMPSS_DIST_SCHEDULE_KIND(Name) .Case(#Name, OSSC_DIST_SCHEDULE_##Name)
#include "clang/Basic/OmpSsKinds.def"
        .Default(OSSC_DIST_SCHEDULE_unknown);
  case OSSC_defaultmap:
    return llvm::StringSwitch<unsigned>(Str)
#define OMPSS_DEFAULTMAP_KIND(Name)                                           \
  .Case(#Name, static_cast<unsigned>(OSSC_DEFAULTMAP_##Name))
#define OMPSS_DEFAULTMAP_MODIFIER(Name)                                       \
  .Case(#Name, static_cast<unsigned>(OSSC_DEFAULTMAP_MODIFIER_##Name))
#include "clang/Basic/OmpSsKinds.def"
        .Default(OSSC_DEFAULTMAP_unknown);
  case OSSC_unknown:
  case OSSC_threadprivate:
  case OSSC_if:
  case OSSC_final:
  case OSSC_num_threads:
  case OSSC_safelen:
  case OSSC_simdlen:
  case OSSC_collapse:
  case OSSC_private:
  case OSSC_firstprivate:
  case OSSC_lastprivate:
  case OSSC_shared:
  case OSSC_reduction:
  case OSSC_task_reduction:
  case OSSC_in_reduction:
  case OSSC_aligned:
  case OSSC_copyin:
  case OSSC_copyprivate:
  case OSSC_ordered:
  case OSSC_nowait:
  case OSSC_untied:
  case OSSC_mergeable:
  case OSSC_flush:
  case OSSC_read:
  case OSSC_write:
  case OSSC_update:
  case OSSC_capture:
  case OSSC_seq_cst:
  case OSSC_device:
  case OSSC_threads:
  case OSSC_simd:
  case OSSC_num_teams:
  case OSSC_thread_limit:
  case OSSC_priority:
  case OSSC_grainsize:
  case OSSC_nogroup:
  case OSSC_num_tasks:
  case OSSC_hint:
  case OSSC_uniform:
  case OSSC_to:
  case OSSC_from:
  case OSSC_use_device_ptr:
  case OSSC_is_device_ptr:
    break;
  }
  llvm_unreachable("Invalid OmpSs simple clause kind");
}

const char *clang::getOmpSsSimpleClauseTypeName(OmpSsClauseKind Kind,
                                                 unsigned Type) {
  switch (Kind) {
  case OSSC_default:
    switch (Type) {
    case OSSC_DEFAULT_unknown:
      return "unknown";
#define OMPSS_DEFAULT_KIND(Name)                                              \
  case OSSC_DEFAULT_##Name:                                                    \
    return #Name;
#include "clang/Basic/OmpSsKinds.def"
    }
    llvm_unreachable("Invalid OmpSs 'default' clause type");
  case OSSC_proc_bind:
    switch (Type) {
    case OSSC_PROC_BIND_unknown:
      return "unknown";
#define OMPSS_PROC_BIND_KIND(Name)                                            \
  case OSSC_PROC_BIND_##Name:                                                  \
    return #Name;
#include "clang/Basic/OmpSsKinds.def"
    }
    llvm_unreachable("Invalid OmpSs 'proc_bind' clause type");
  case OSSC_schedule:
    switch (Type) {
    case OSSC_SCHEDULE_unknown:
    case OSSC_SCHEDULE_MODIFIER_last:
      return "unknown";
#define OMPSS_SCHEDULE_KIND(Name)                                             \
    case OSSC_SCHEDULE_##Name:                                                 \
      return #Name;
#define OMPSS_SCHEDULE_MODIFIER(Name)                                         \
    case OSSC_SCHEDULE_MODIFIER_##Name:                                        \
      return #Name;
#include "clang/Basic/OmpSsKinds.def"
    }
    llvm_unreachable("Invalid OmpSs 'schedule' clause type");
  case OSSC_depend:
    switch (Type) {
    case OSSC_DEPEND_unknown:
      return "unknown";
#define OMPSS_DEPEND_KIND(Name)                                             \
  case OSSC_DEPEND_##Name:                                                   \
    return #Name;
#include "clang/Basic/OmpSsKinds.def"
    }
    llvm_unreachable("Invalid OmpSs 'depend' clause type");
  case OSSC_linear:
    switch (Type) {
    case OSSC_LINEAR_unknown:
      return "unknown";
#define OMPSS_LINEAR_KIND(Name)                                             \
  case OSSC_LINEAR_##Name:                                                   \
    return #Name;
#include "clang/Basic/OmpSsKinds.def"
    }
    llvm_unreachable("Invalid OmpSs 'linear' clause type");
  case OSSC_map:
    switch (Type) {
    case OSSC_MAP_unknown:
      return "unknown";
#define OMPSS_MAP_KIND(Name)                                                \
  case OSSC_MAP_##Name:                                                      \
    return #Name;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    llvm_unreachable("Invalid OmpSs 'map' clause type");
  case OSSC_dist_schedule:
    switch (Type) {
    case OSSC_DIST_SCHEDULE_unknown:
      return "unknown";
#define OMPSS_DIST_SCHEDULE_KIND(Name)                                      \
  case OSSC_DIST_SCHEDULE_##Name:                                            \
    return #Name;
#include "clang/Basic/OmpSsKinds.def"
    }
    llvm_unreachable("Invalid OmpSs 'dist_schedule' clause type");
  case OSSC_defaultmap:
    switch (Type) {
    case OSSC_DEFAULTMAP_unknown:
    case OSSC_DEFAULTMAP_MODIFIER_last:
      return "unknown";
#define OMPSS_DEFAULTMAP_KIND(Name)                                         \
    case OSSC_DEFAULTMAP_##Name:                                             \
      return #Name;
#define OMPSS_DEFAULTMAP_MODIFIER(Name)                                     \
    case OSSC_DEFAULTMAP_MODIFIER_##Name:                                    \
      return #Name;
#include "clang/Basic/OmpSsKinds.def"
    }
    llvm_unreachable("Invalid OmpSs 'schedule' clause type");
  case OSSC_unknown:
  case OSSC_threadprivate:
  case OSSC_if:
  case OSSC_final:
  case OSSC_num_threads:
  case OSSC_safelen:
  case OSSC_simdlen:
  case OSSC_collapse:
  case OSSC_private:
  case OSSC_firstprivate:
  case OSSC_lastprivate:
  case OSSC_shared:
  case OSSC_reduction:
  case OSSC_task_reduction:
  case OSSC_in_reduction:
  case OSSC_aligned:
  case OSSC_copyin:
  case OSSC_copyprivate:
  case OSSC_ordered:
  case OSSC_nowait:
  case OSSC_untied:
  case OSSC_mergeable:
  case OSSC_flush:
  case OSSC_read:
  case OSSC_write:
  case OSSC_update:
  case OSSC_capture:
  case OSSC_seq_cst:
  case OSSC_device:
  case OSSC_threads:
  case OSSC_simd:
  case OSSC_num_teams:
  case OSSC_thread_limit:
  case OSSC_priority:
  case OSSC_grainsize:
  case OSSC_nogroup:
  case OSSC_num_tasks:
  case OSSC_hint:
  case OSSC_uniform:
  case OSSC_to:
  case OSSC_from:
  case OSSC_use_device_ptr:
  case OSSC_is_device_ptr:
    break;
  }
  llvm_unreachable("Invalid OmpSs simple clause kind");
}

bool clang::isAllowedClauseForDirective(OmpSsDirectiveKind DKind,
                                        OmpSsClauseKind CKind) {
  assert(DKind <= OSSD_unknown);
  assert(CKind <= OSSC_unknown);
  switch (DKind) {
  case OSSD_parallel:
    switch (CKind) {
#define OMPSS_PARALLEL_CLAUSE(Name)                                           \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_simd:
    switch (CKind) {
#define OMPSS_SIMD_CLAUSE(Name)                                               \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_for:
    switch (CKind) {
#define OMPSS_FOR_CLAUSE(Name)                                                \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_for_simd:
    switch (CKind) {
#define OMPSS_FOR_SIMD_CLAUSE(Name)                                           \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_sections:
    switch (CKind) {
#define OMPSS_SECTIONS_CLAUSE(Name)                                           \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_single:
    switch (CKind) {
#define OMPSS_SINGLE_CLAUSE(Name)                                             \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_parallel_for:
    switch (CKind) {
#define OMPSS_PARALLEL_FOR_CLAUSE(Name)                                       \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_parallel_for_simd:
    switch (CKind) {
#define OMPSS_PARALLEL_FOR_SIMD_CLAUSE(Name)                                  \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_parallel_sections:
    switch (CKind) {
#define OMPSS_PARALLEL_SECTIONS_CLAUSE(Name)                                  \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_task:
    switch (CKind) {
#define OMPSS_TASK_CLAUSE(Name)                                               \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_flush:
    return CKind == OSSC_flush;
    break;
  case OSSD_atomic:
    switch (CKind) {
#define OMPSS_ATOMIC_CLAUSE(Name)                                             \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target:
    switch (CKind) {
#define OMPSS_TARGET_CLAUSE(Name)                                             \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_data:
    switch (CKind) {
#define OMPSS_TARGET_DATA_CLAUSE(Name)                                        \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_enter_data:
    switch (CKind) {
#define OMPSS_TARGET_ENTER_DATA_CLAUSE(Name)                                  \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_exit_data:
    switch (CKind) {
#define OMPSS_TARGET_EXIT_DATA_CLAUSE(Name)                                   \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_parallel:
    switch (CKind) {
#define OMPSS_TARGET_PARALLEL_CLAUSE(Name)                                    \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_parallel_for:
    switch (CKind) {
#define OMPSS_TARGET_PARALLEL_FOR_CLAUSE(Name)                                \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_update:
    switch (CKind) {
#define OMPSS_TARGET_UPDATE_CLAUSE(Name)                                      \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_teams:
    switch (CKind) {
#define OMPSS_TEAMS_CLAUSE(Name)                                              \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_declare_simd:
    break;
  case OSSD_cancel:
    switch (CKind) {
#define OMPSS_CANCEL_CLAUSE(Name)                                             \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_ordered:
    switch (CKind) {
#define OMPSS_ORDERED_CLAUSE(Name)                                            \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_taskloop:
    switch (CKind) {
#define OMPSS_TASKLOOP_CLAUSE(Name)                                           \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_taskloop_simd:
    switch (CKind) {
#define OMPSS_TASKLOOP_SIMD_CLAUSE(Name)                                      \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_critical:
    switch (CKind) {
#define OMPSS_CRITICAL_CLAUSE(Name)                                           \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_distribute:
    switch (CKind) {
#define OMPSS_DISTRIBUTE_CLAUSE(Name)                                         \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_distribute_parallel_for:
    switch (CKind) {
#define OMPSS_DISTRIBUTE_PARALLEL_FOR_CLAUSE(Name)                            \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_distribute_parallel_for_simd:
    switch (CKind) {
#define OMPSS_DISTRIBUTE_PARALLEL_FOR_SIMD_CLAUSE(Name)                       \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_distribute_simd:
    switch (CKind) {
#define OMPSS_DISTRIBUTE_SIMD_CLAUSE(Name)                                    \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_parallel_for_simd:
    switch (CKind) {
#define OMPSS_TARGET_PARALLEL_FOR_SIMD_CLAUSE(Name)                           \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_simd:
    switch (CKind) {
#define OMPSS_TARGET_SIMD_CLAUSE(Name)                                        \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_teams_distribute:
    switch (CKind) {
#define OMPSS_TEAMS_DISTRIBUTE_CLAUSE(Name)                                   \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_teams_distribute_simd:
    switch (CKind) {
#define OMPSS_TEAMS_DISTRIBUTE_SIMD_CLAUSE(Name)                              \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_teams_distribute_parallel_for_simd:
    switch (CKind) {
#define OMPSS_TEAMS_DISTRIBUTE_PARALLEL_FOR_SIMD_CLAUSE(Name)                 \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_teams_distribute_parallel_for:
    switch (CKind) {
#define OMPSS_TEAMS_DISTRIBUTE_PARALLEL_FOR_CLAUSE(Name)                      \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_teams:
    switch (CKind) {
#define OMPSS_TARGET_TEAMS_CLAUSE(Name)                                       \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_teams_distribute:
    switch (CKind) {
#define OMPSS_TARGET_TEAMS_DISTRIBUTE_CLAUSE(Name)                            \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_teams_distribute_parallel_for:
    switch (CKind) {
#define OMPSS_TARGET_TEAMS_DISTRIBUTE_PARALLEL_FOR_CLAUSE(Name)               \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_teams_distribute_parallel_for_simd:
    switch (CKind) {
#define OMPSS_TARGET_TEAMS_DISTRIBUTE_PARALLEL_FOR_SIMD_CLAUSE(Name)          \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_target_teams_distribute_simd:
    switch (CKind) {
#define OMPSS_TARGET_TEAMS_DISTRIBUTE_SIMD_CLAUSE(Name)                       \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_taskgroup:
    switch (CKind) {
#define OMPSS_TASKGROUP_CLAUSE(Name)                                          \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_declare_target:
  case OSSD_end_declare_target:
  case OSSD_unknown:
  case OSSD_threadprivate:
  case OSSD_section:
  case OSSD_master:
  case OSSD_taskyield:
  case OSSD_barrier:
  case OSSD_taskwait:
  case OSSD_cancellation_point:
  case OSSD_declare_reduction:
    break;
  }
  return false;
}

bool clang::isOmpSsLoopDirective(OmpSsDirectiveKind DKind) {
  return DKind == OSSD_simd || DKind == OSSD_for || DKind == OSSD_for_simd ||
         DKind == OSSD_parallel_for || DKind == OSSD_parallel_for_simd ||
         DKind == OSSD_taskloop || DKind == OSSD_taskloop_simd ||
         DKind == OSSD_distribute || DKind == OSSD_target_parallel_for ||
         DKind == OSSD_distribute_parallel_for ||
         DKind == OSSD_distribute_parallel_for_simd ||
         DKind == OSSD_distribute_simd ||
         DKind == OSSD_target_parallel_for_simd || DKind == OSSD_target_simd ||
         DKind == OSSD_teams_distribute ||
         DKind == OSSD_teams_distribute_simd ||
         DKind == OSSD_teams_distribute_parallel_for_simd ||
         DKind == OSSD_teams_distribute_parallel_for ||
         DKind == OSSD_target_teams_distribute ||
         DKind == OSSD_target_teams_distribute_parallel_for ||
         DKind == OSSD_target_teams_distribute_parallel_for_simd ||
         DKind == OSSD_target_teams_distribute_simd;
}

bool clang::isOmpSsWorksharingDirective(OmpSsDirectiveKind DKind) {
  return DKind == OSSD_for || DKind == OSSD_for_simd ||
         DKind == OSSD_sections || DKind == OSSD_section ||
         DKind == OSSD_single || DKind == OSSD_parallel_for ||
         DKind == OSSD_parallel_for_simd || DKind == OSSD_parallel_sections ||
         DKind == OSSD_target_parallel_for ||
         DKind == OSSD_distribute_parallel_for ||
         DKind == OSSD_distribute_parallel_for_simd ||
         DKind == OSSD_target_parallel_for_simd ||
         DKind == OSSD_teams_distribute_parallel_for_simd ||
         DKind == OSSD_teams_distribute_parallel_for ||
         DKind == OSSD_target_teams_distribute_parallel_for ||
         DKind == OSSD_target_teams_distribute_parallel_for_simd;
}

bool clang::isOmpSsTaskLoopDirective(OmpSsDirectiveKind DKind) {
  return DKind == OSSD_taskloop || DKind == OSSD_taskloop_simd;
}

bool clang::isOmpSsParallelDirective(OmpSsDirectiveKind DKind) {
  return DKind == OSSD_parallel || DKind == OSSD_parallel_for ||
         DKind == OSSD_parallel_for_simd || DKind == OSSD_parallel_sections ||
         DKind == OSSD_target_parallel || DKind == OSSD_target_parallel_for ||
         DKind == OSSD_distribute_parallel_for ||
         DKind == OSSD_distribute_parallel_for_simd ||
         DKind == OSSD_target_parallel_for_simd ||
         DKind == OSSD_teams_distribute_parallel_for ||
         DKind == OSSD_teams_distribute_parallel_for_simd ||
         DKind == OSSD_target_teams_distribute_parallel_for ||
         DKind == OSSD_target_teams_distribute_parallel_for_simd;
}

bool clang::isOmpSsTargetExecutionDirective(OmpSsDirectiveKind DKind) {
  return DKind == OSSD_target || DKind == OSSD_target_parallel ||
         DKind == OSSD_target_parallel_for ||
         DKind == OSSD_target_parallel_for_simd || DKind == OSSD_target_simd ||
         DKind == OSSD_target_teams || DKind == OSSD_target_teams_distribute ||
         DKind == OSSD_target_teams_distribute_parallel_for ||
         DKind == OSSD_target_teams_distribute_parallel_for_simd ||
         DKind == OSSD_target_teams_distribute_simd;
}

bool clang::isOmpSsTargetDataManagementDirective(OmpSsDirectiveKind DKind) {
  return DKind == OSSD_target_data || DKind == OSSD_target_enter_data ||
         DKind == OSSD_target_exit_data || DKind == OSSD_target_update;
}

bool clang::isOmpSsNestingTeamsDirective(OmpSsDirectiveKind DKind) {
  return DKind == OSSD_teams || DKind == OSSD_teams_distribute ||
         DKind == OSSD_teams_distribute_simd ||
         DKind == OSSD_teams_distribute_parallel_for_simd ||
         DKind == OSSD_teams_distribute_parallel_for;
}

bool clang::isOmpSsTeamsDirective(OmpSsDirectiveKind DKind) {
  return isOmpSsNestingTeamsDirective(DKind) ||
         DKind == OSSD_target_teams || DKind == OSSD_target_teams_distribute ||
         DKind == OSSD_target_teams_distribute_parallel_for ||
         DKind == OSSD_target_teams_distribute_parallel_for_simd ||
         DKind == OSSD_target_teams_distribute_simd;
}

bool clang::isOmpSsSimdDirective(OmpSsDirectiveKind DKind) {
  return DKind == OSSD_simd || DKind == OSSD_for_simd ||
         DKind == OSSD_parallel_for_simd || DKind == OSSD_taskloop_simd ||
         DKind == OSSD_distribute_parallel_for_simd ||
         DKind == OSSD_distribute_simd || DKind == OSSD_target_simd ||
         DKind == OSSD_teams_distribute_simd ||
         DKind == OSSD_teams_distribute_parallel_for_simd ||
         DKind == OSSD_target_teams_distribute_parallel_for_simd ||
         DKind == OSSD_target_teams_distribute_simd ||
         DKind == OSSD_target_parallel_for_simd;
}

bool clang::isOmpSsNestingDistributeDirective(OmpSsDirectiveKind Kind) {
  return Kind == OSSD_distribute || Kind == OSSD_distribute_parallel_for ||
         Kind == OSSD_distribute_parallel_for_simd ||
         Kind == OSSD_distribute_simd;
  // TODO add next directives.
}

bool clang::isOmpSsDistributeDirective(OmpSsDirectiveKind Kind) {
  return isOmpSsNestingDistributeDirective(Kind) ||
         Kind == OSSD_teams_distribute || Kind == OSSD_teams_distribute_simd ||
         Kind == OSSD_teams_distribute_parallel_for_simd ||
         Kind == OSSD_teams_distribute_parallel_for ||
         Kind == OSSD_target_teams_distribute ||
         Kind == OSSD_target_teams_distribute_parallel_for ||
         Kind == OSSD_target_teams_distribute_parallel_for_simd ||
         Kind == OSSD_target_teams_distribute_simd;
}

bool clang::isOmpSsPrivate(OmpSsClauseKind Kind) {
  return Kind == OSSC_private || Kind == OSSC_firstprivate ||
         Kind == OSSC_lastprivate || Kind == OSSC_linear ||
         Kind == OSSC_reduction || Kind == OSSC_task_reduction ||
         Kind == OSSC_in_reduction; // TODO add next clauses like 'reduction'.
}

bool clang::isOmpSsThreadPrivate(OmpSsClauseKind Kind) {
  return Kind == OSSC_threadprivate || Kind == OSSC_copyin;
}

bool clang::isOmpSsTaskingDirective(OmpSsDirectiveKind Kind) {
  return Kind == OSSD_task || isOmpSsTaskLoopDirective(Kind);
}

bool clang::isOmpSsLoopBoundSharingDirective(OmpSsDirectiveKind Kind) {
  return Kind == OSSD_distribute_parallel_for ||
         Kind == OSSD_distribute_parallel_for_simd ||
         Kind == OSSD_teams_distribute_parallel_for_simd ||
         Kind == OSSD_teams_distribute_parallel_for ||
         Kind == OSSD_target_teams_distribute_parallel_for ||
         Kind == OSSD_target_teams_distribute_parallel_for_simd;
}

void clang::getOmpSsCaptureRegions(
    SmallVectorImpl<OmpSsDirectiveKind> &CaptureRegions,
    OmpSsDirectiveKind DKind) {
  assert(DKind <= OSSD_unknown);
  switch (DKind) {
  case OSSD_parallel:
  case OSSD_parallel_for:
  case OSSD_parallel_for_simd:
  case OSSD_parallel_sections:
  case OSSD_distribute_parallel_for:
  case OSSD_distribute_parallel_for_simd:
    CaptureRegions.push_back(OSSD_parallel);
    break;
  case OSSD_target_teams:
  case OSSD_target_teams_distribute:
  case OSSD_target_teams_distribute_simd:
    CaptureRegions.push_back(OSSD_task);
    CaptureRegions.push_back(OSSD_target);
    CaptureRegions.push_back(OSSD_teams);
    break;
  case OSSD_teams:
  case OSSD_teams_distribute:
  case OSSD_teams_distribute_simd:
    CaptureRegions.push_back(OSSD_teams);
    break;
  case OSSD_target:
  case OSSD_target_simd:
    CaptureRegions.push_back(OSSD_task);
    CaptureRegions.push_back(OSSD_target);
    break;
  case OSSD_teams_distribute_parallel_for:
  case OSSD_teams_distribute_parallel_for_simd:
    CaptureRegions.push_back(OSSD_teams);
    CaptureRegions.push_back(OSSD_parallel);
    break;
  case OSSD_target_parallel:
  case OSSD_target_parallel_for:
  case OSSD_target_parallel_for_simd:
    CaptureRegions.push_back(OSSD_task);
    CaptureRegions.push_back(OSSD_target);
    CaptureRegions.push_back(OSSD_parallel);
    break;
  case OSSD_task:
  case OSSD_target_enter_data:
  case OSSD_target_exit_data:
  case OSSD_target_update:
    CaptureRegions.push_back(OSSD_task);
    break;
  case OSSD_taskloop:
  case OSSD_taskloop_simd:
    CaptureRegions.push_back(OSSD_taskloop);
    break;
  case OSSD_target_teams_distribute_parallel_for:
  case OSSD_target_teams_distribute_parallel_for_simd:
    CaptureRegions.push_back(OSSD_task);
    CaptureRegions.push_back(OSSD_target);
    CaptureRegions.push_back(OSSD_teams);
    CaptureRegions.push_back(OSSD_parallel);
    break;
  case OSSD_simd:
  case OSSD_for:
  case OSSD_for_simd:
  case OSSD_sections:
  case OSSD_section:
  case OSSD_single:
  case OSSD_master:
  case OSSD_critical:
  case OSSD_taskgroup:
  case OSSD_distribute:
  case OSSD_ordered:
  case OSSD_atomic:
  case OSSD_target_data:
  case OSSD_distribute_simd:
    CaptureRegions.push_back(OSSD_unknown);
    break;
  case OSSD_threadprivate:
  case OSSD_taskyield:
  case OSSD_barrier:
  case OSSD_taskwait:
  case OSSD_cancellation_point:
  case OSSD_cancel:
  case OSSD_flush:
  case OSSD_declare_reduction:
  case OSSD_declare_simd:
  case OSSD_declare_target:
  case OSSD_end_declare_target:
    llvm_unreachable("OmpSs Directive is not allowed");
  case OSSD_unknown:
    llvm_unreachable("Unknown OmpSs directive");
  }
}
