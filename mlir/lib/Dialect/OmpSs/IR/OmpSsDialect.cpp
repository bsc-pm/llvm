//===- OmpSsDialect.cpp - MLIR Dialect for OmpSs implementation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OmpSs dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OmpSs/OmpSsDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cstddef>

#include "mlir/Dialect/OmpSs/OmpSsOpsDialect.cpp.inc"
#include "mlir/Dialect/OmpSs/OmpSsOpsEnums.cpp.inc"
#include "mlir/Dialect/OmpSs/OmpSsOpsInterfaces.cpp.inc"

using namespace mlir;
using namespace mlir::oss;

void OmpSsDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OmpSs/OmpSsOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/OmpSs/OmpSsOpsAttributes.cpp.inc"
      >();
}

// TODO: remove functs

/// Parse a list of operands with types.
///
/// operand-and-type-list ::= `(` ssa-id-and-type-list `)`
/// ssa-id-and-type-list ::= ssa-id-and-type |
///                          ssa-id-and-type `,` ssa-id-and-type-list
/// ssa-id-and-type ::= ssa-id `:` type
static ParseResult
parseOperandAndTypeList(OpAsmParser &parser,
                        SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                        SmallVectorImpl<Type> &types) {
  if (parser.parseLParen())
    return failure();

  do {
    OpAsmParser::UnresolvedOperand operand;
    Type type;
    if (parser.parseOperand(operand) || parser.parseColonType(type))
      return failure();
    operands.push_back(operand);
    types.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return failure();

  return success();
}

static LogicalResult verifyTaskOp(TaskOp op) {
  return success();
}

static void printTaskOp(OpAsmPrinter &p, TaskOp op) {
  p << "oss.task";

  if (auto ifCond = op.getIfExprVar())
    p << " if(" << ifCond << " : " << ifCond.getType() << ")";

  if (auto finalCond = op.getFinalExprVar())
    p << " final(" << finalCond << " : " << finalCond.getType() << ")";

  if (auto costCond = op.getCostExprVar())
    p << " cost(" << costCond << " : " << costCond.getType() << ")";

  if (auto priorityCond = op.getPriorityExprVar())
    p << " priority(" << priorityCond << " : " << priorityCond.getType() << ")";

  if (auto def = op.getDefaultVal())
    p << " default(" << stringifyClauseDefault(def.value()) << ")";

  // Print private, firstprivate, shared and copyin parameters
  auto printDataVars = [&p](StringRef name, OperandRange vars) {
    if (vars.size()) {
      p << " " << name << "(";
      for (unsigned i = 0; i < vars.size(); ++i) {
        std::string separator = i == vars.size() - 1 ? ")" : ", ";
        p << vars[i] << " : " << vars[i].getType() << separator;
      }
    }
  };

  printDataVars("private", op.getPrivateVars());
  printDataVars("firstprivate", op.getFirstprivateVars());
  printDataVars("shared", op.getSharedVars());

  p.printRegion(op.getRegion());
}

/// Emit an error if the same clause is present more than once on an operation.
static ParseResult allowedOnce(OpAsmParser &parser, llvm::StringRef clause,
                               llvm::StringRef operation) {
  return parser.emitError(parser.getNameLoc())
         << " at most one " << clause << " clause can appear on the "
         << operation << " operation";
}

/// Parses a parallel operation.
///
/// operation ::= `omp.parallel` clause-list
/// clause-list ::= clause | clause clause-list
/// clause ::= if | final | cost | priority | default | private | firstprivate |
///            shared
/// if ::= `if` `(` ssa-id `)`
/// final ::= `final` `(` ssa-id `)`
/// cost ::= `cost` `(` ssa-id `)`
/// priority ::= `priority` `(` ssa-id `)`
/// default ::= `default` `(` (`private` | `firstprivate` | `shared` | `none`)
/// private ::= `private` operand-and-type-list
/// firstprivate ::= `firstprivate` operand-and-type-list
/// shared ::= `shared` operand-and-type-list
///
/// Note that each clause can only appear once in the clase-list.
static ParseResult parseTaskOp(OpAsmParser &parser,
                                   OperationState &result) {
  std::pair<OpAsmParser::UnresolvedOperand, Type> ifCond;
  std::pair<OpAsmParser::UnresolvedOperand, Type> finalCond;
  std::pair<OpAsmParser::UnresolvedOperand, Type> costCond;
  std::pair<OpAsmParser::UnresolvedOperand, Type> priorityCond;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> privates;
  SmallVector<Type, 4> privateTypes;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> firstprivates;
  SmallVector<Type, 4> firstprivateTypes;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> shareds;
  SmallVector<Type, 4> sharedTypes;
  std::array<int, 7> segments{0, 0, 0, 0, 0, 0, 0};
  llvm::StringRef keyword;
  bool defaultVal = false;

  const int ifClausePos           = 0;
  const int finalClausePos        = 1;
  const int costClausePos         = 2;
  const int priorityClausePos     = 3;
  const int privateClausePos      = 4;
  const int firstprivateClausePos = 5;
  const int sharedClausePos       = 6;
  const llvm::StringRef opName = result.name.getStringRef();

  while (succeeded(parser.parseOptionalKeyword(&keyword))) {
    if (keyword == "if") {
      // Fail if there was already another if condition
      if (segments[ifClausePos])
        return allowedOnce(parser, "if", opName);
      if (parser.parseLParen() || parser.parseOperand(ifCond.first) ||
          parser.parseColonType(ifCond.second) || parser.parseRParen())
        return failure();
      segments[ifClausePos] = 1;
    } else if (keyword == "final") {
      // Fail final there was already another final condition
      if (segments[finalClausePos])
        return allowedOnce(parser, "final", opName);
      if (parser.parseLParen() || parser.parseOperand(finalCond.first) ||
          parser.parseColonType(finalCond.second) || parser.parseRParen())
        return failure();
      segments[finalClausePos] = 1;
    } else if (keyword == "cost") {
      // Fail cost there was already another cost condition
      if (segments[costClausePos])
        return allowedOnce(parser, "cost", opName);
      if (parser.parseLParen() || parser.parseOperand(costCond.first) ||
          parser.parseColonType(costCond.second) || parser.parseRParen())
        return failure();
      segments[costClausePos] = 1;
    } else if (keyword == "priority") {
      // Fail priority there was already another priority condition
      if (segments[priorityClausePos])
        return allowedOnce(parser, "priority", opName);
      if (parser.parseLParen() || parser.parseOperand(priorityCond.first) ||
          parser.parseColonType(priorityCond.second) || parser.parseRParen())
        return failure();
      segments[priorityClausePos] = 1;
    } else if (keyword == "private") {
      // fail if there was already another private clause
      if (segments[privateClausePos])
        return allowedOnce(parser, "private", opName);
      if (parseOperandAndTypeList(parser, privates, privateTypes))
        return failure();
      segments[privateClausePos] = privates.size();
    } else if (keyword == "firstprivate") {
      // fail if there was already another firstprivate clause
      if (segments[firstprivateClausePos])
        return allowedOnce(parser, "firstprivate", opName);
      if (parseOperandAndTypeList(parser, firstprivates, firstprivateTypes))
        return failure();
      segments[firstprivateClausePos] = firstprivates.size();
    } else if (keyword == "shared") {
      // fail if there was already another shared clause
      if (segments[sharedClausePos])
        return allowedOnce(parser, "shared", opName);
      if (parseOperandAndTypeList(parser, shareds, sharedTypes))
        return failure();
      segments[sharedClausePos] = shareds.size();
    } else if (keyword == "default") {
      // fail if there was already another default clause
      if (defaultVal)
        return allowedOnce(parser, "default", opName);
      defaultVal = true;
      llvm::StringRef defval;
      if (parser.parseLParen() || parser.parseKeyword(&defval) ||
          parser.parseRParen())
        return failure();
      llvm::SmallString<16> attrval;
      // The def prefix is required for the attribute as "private" is a keyword
      // in C++
      attrval += "def";
      attrval += defval;
      auto attr = parser.getBuilder().getStringAttr(attrval);
      result.addAttribute("default_val", attr);
    } else {
      return parser.emitError(parser.getNameLoc())
             << keyword << " is not a valid clause for the " << opName
             << " operation";
    }
  }

  // Add if parameter
  if (segments[ifClausePos] &&
      parser.resolveOperand(ifCond.first, ifCond.second, result.operands))
    return failure();

  // Add final parameter
  if (segments[finalClausePos] &&
      parser.resolveOperand(finalCond.first, finalCond.second, result.operands))
    return failure();

  // Add cost parameter
  if (segments[costClausePos] &&
      parser.resolveOperand(costCond.first, costCond.second, result.operands))
    return failure();

  // Add priority parameter
  if (segments[priorityClausePos] &&
      parser.resolveOperand(priorityCond.first, priorityCond.second, result.operands))
    return failure();

  // Add private parameters
  if (segments[privateClausePos] &&
      parser.resolveOperands(privates, privateTypes, privates[0].location,
                             result.operands))
    return failure();

  // Add firstprivate parameters
  if (segments[firstprivateClausePos] &&
      parser.resolveOperands(firstprivates, firstprivateTypes,
                             firstprivates[0].location, result.operands))
    return failure();

  // Add shared parameters
  if (segments[sharedClausePos] &&
      parser.resolveOperands(shareds, sharedTypes, shareds[0].location,
                             result.operands))
    return failure();

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(segments));

  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();
  return success();
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OmpSs/OmpSsOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/OmpSs/OmpSsOps.cpp.inc"
