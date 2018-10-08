#!/usr/bin/env python2
import sys
import os
import logging
import copy
import re

class Field:
    pass

class FixedBits(Field):
    def __init__(self, text):
        self.text = text

    def bits(self):
        return len(self.text)

    @staticmethod
    def make(text):
        return FixedBits(text)

    def __str__(self):
        return "0b{}".format(self.text)

class VectorMask(Field):
    def __init__(self):
        pass

    def bits(self):
        return 2

    @staticmethod
    def make(text):
        return VectorMask()

    def __str__(self):
        return "vm"

class SlicedImmediate(Field):
    def __init__(self, upper, lower):
        assert(upper > lower)
        self._upper = upper
        self._lower = lower
        self.num_bits = upper - lower + 1

    def bits(self):
        return self.num_bits

    def lower(self):
        return self._lower

    def upper(self):
        return self._upper

    @staticmethod
    def make(text):
        # imm[upper:lower]
        t = text[4:][:-1].split(":")
        assert(len(t) == 2)
        return SlicedImmediate(int(t[0]), int(t[1]))

    def __str__(self):
        return "imm[{}:{}]".format(self._upper, self._lower)

class Register(Field):
    ## Kind
    GPR = 1
    VR = 2
    FPR = 3

    ## Direction
    IN = 1
    OUT = 2
    INOUT = 3

    GPR_ABI_NAMES = ["zero,", "ra", "sp", "gp", "tp"] + \
                ["t{}".format(x) for x in range(0, 3)] + \
                ["s{}".format(x) for x in range(0, 2)] + \
                ["a{}".format(x) for x in range(0, 8)] + \
                ["s{}".format(x) for x in range(2, 12)] + \
                ["t{}".format(x) for x in range(3, 7)]

    FPR_ABI_NAMES = \
                ["ft{}".format(x) for x in range(0, 8)] + \
                ["fs{}".format(x) for x in range(0, 2)] + \
                ["fa{}".format(x) for x in range(0, 8)] + \
                ["fs{}".format(x) for x in range(2, 12)] + \
                ["ft{}".format(x) for x in range(8, 12)]

    def __init__(self, name, kind, direction, index):
        self.name = name
        self.kind = kind
        self.direction = direction
        self.index = index
        self.scale = 1 # 1 x SEW

    def __str__(self):
        return self.name

    def bits(self):
        return 5

    @staticmethod
    def make(text):
        kind = None
        if text[0] == "r":
            kind = Register.GPR
        elif text[0] == "v":
            kind = Register.VR
        elif text[0] == "f":
            kind = Register.FPR
        if not kind:
            raise Exception("Uknown register kind '{}'".format(text))

        ## FIXME: Some operands are destructive (e.g. VINSERT)
        ## FIXME: mark them as vx/rx in the spec
        direction = None
        if text[1] == "s":
            direction = Register.IN
        elif text[1] == "d":
            direction = Register.OUT
        if not direction:
            raise Exception("Uknown register direction '{}'".format(text))

        index = 0
        if text[1] == "s":
            index = int(text[2:])

        return Register(text, kind, direction, index)

class Instruction:
    def __init__(self, name, fields, operand_order):
        self.name = name
        self.fields = fields
        self.operand_order = operand_order
        pass

    def __str__(self):
        return "{} = [{}], <{}>".format(self.name,
                                  ", ".join([str(x) for x in self.fields]),
                                  ",".join(self.operand_order))

    def immediate_width(self):
        w = 0
        l = 32
        for f in self.fields:
            if isinstance(f, SlicedImmediate):
                w = max(f.upper(), w)
                l = min(f.lower(), l)
        assert(l == 0)
        return w + 1

class InstructionParser:
    INSTRUCTION = re.compile("^[A-Z0-9.]+$")
    BITS = re.compile("^[01]+$")
    GPR = re.compile("^(rs[0-9]+)|(rd)$")
    FPR = re.compile("^(fs[0-9]+)|(fd)$")
    VR = re.compile("^(vs[0-9]+)|(vd)$")
    IMM = re.compile("^imm\[[0-9]+:[0-9]+\]$")
    MASK = re.compile("^vm$")

    def __init__(self):
        self.instructions = []

    def is_error(self, m, s):
        return False if m.match(s) else True

    def parse_mnemonic(self, name):
        return (name, self.is_error(InstructionParser.INSTRUCTION, name))

    def parse_field(self, text):
        field_types = [
               (InstructionParser.BITS, FixedBits.make),
               (InstructionParser.GPR, Register.make),
               (InstructionParser.VR, Register.make),
               (InstructionParser.MASK, VectorMask.make),
               (InstructionParser.IMM, SlicedImmediate.make),
               ]
        for (m, ctor) in field_types:
            if self.is_error(m, text):
                continue
            return (ctor(text), False)

        # Error
        return (text, True)

    @staticmethod
    def fix_operands(instr):
        iname = instr.name.lower()
        # Narrowing instructions are special. Handle them first
        opcode_parts  = iname.split(".")

        # Special case for floating registers that they may appear in the input
        # as rsX
        if iname.endswith(".vf") or iname.endswith(".wf"):
            for (i, op) in enumerate(instr.fields):
                if isinstance(op, Register) and op.name == "rs1":
                    op.name = "fs1"
                    op.kind = Register.FPR
                    break
            for (i, op) in enumerate(instr.operand_order):
                if op == "rs1":
                    instr.operand_order[i] = "fs1"

        # Special cases
        if iname == "vfmv.s.f":
            for (i, op) in enumerate(instr.fields):
                if isinstance(op, Register) and op.name == "rs1":
                    op.name = "fs1"
                    op.kind = Register.FPR
                    break
            for (i, op) in enumerate(instr.operand_order):
                if op == "rs1":
                    instr.operand_order[i] = "fs1"

        if iname == "vfmv.f.s":
            for (i, op) in enumerate(instr.fields):
                if isinstance(op, Register) and op.name == "rd":
                    op.name = "fd"
                    op.kind = Register.FPR
                    break
            for (i, op) in enumerate(instr.operand_order):
                if op == "rd":
                    instr.operand_order[i] = "fd"

    def parse_input(self, input_filename):
        input_file = open(input_filename, "r")
        line_number = 0
        for line in input_file:
            line_number += 1
            seen_error = False
            # Skip comments and empty lines
            if line.strip() == "" or line.lstrip()[0] == "#":
                continue

            fields = []
            str_fields = line.split("|")
            assert(str_fields)

            # Instruction name
            (instruction_name, seen_error) = \
                    self.parse_mnemonic(str_fields[0].strip())
            if seen_error:
                logging.warn("ignoring line {} because instruction "
                             "mnemonic '{}' is invalid"
                             .format(line_number, str_fields[-1].strip()))
                continue
            str_fields.pop(0)

            # Order of operands
            operand_order = str_fields[0].split(",")
            str_fields.pop(0)

            # Other fields
            field_number = 0
            for str_field in str_fields:
                field_number += 1
                current_str_field = str_field.strip()
                # Allow empty fields
                if current_str_field == "":
                    continue
                (field, seen_error) = self.parse_field(current_str_field)
                if seen_error:
                    logging.warn("ignoring line {} because field {} with value '{}' "
                                 "is invalid".format(line_number,
                                                     field_number,
                                                     str_field))
                    break
                fields.append(field)
            if seen_error:
                continue
            i = Instruction(instruction_name, fields, operand_order)
            InstructionParser.fix_operands(i)
            self.instructions.append(i)

class Context:
    next_gpr = 1
    next_fpr = 0
    next_vr = 0
    next_imm = 0

    @staticmethod
    def reset():
        Context.next_gpr = 1
        Context.next_fpr = 0
        Context.next_vr = 0
        Context.next_imm = 0

    @staticmethod
    def get_next_gpr():
        res = Context.next_gpr
        # Avoid X0
        Context.next_gpr = 1 + (Context.next_gpr + 1) % 31
        return res

    @staticmethod
    def get_next_fpr():
        res = Context.next_fpr
        Context.next_fpr = (Context.next_fpr + 1) % 31
        return res

    @staticmethod
    def get_next_vr():
        res = Context.next_vr
        Context.next_vr = (Context.next_vr + 1) % 32
        return res

    @staticmethod
    def get_next_vr_modulo(m):
        res = Context.get_next_vr()
        while res % m != 0:
            res = Context.get_next_vr()
        return res

    @staticmethod
    def get_next_imm():
        res = Context.next_imm
        Context.next_imm = Context.next_imm + 1
        return res

    @staticmethod
    def get_next_uimm(width):
        assert(width > 0)
        x = Context.get_next_imm()
        res = x % 2**width
        return res

    @staticmethod
    def get_next_simm(width):
        assert(width > 1)
        res = Context.get_next_uimm(width)
        w = 2**width
        cutoff = w >> 1
        if res >= cutoff:
            res = res - w
        return res

    @staticmethod
    def get_next_simm_as_imm(width):
        res = Context.get_next_simm(width)
        if res < 0:
            res = (2 ** width) + res
        return res

def generate_instruction_tests(i):
    # We need to gather the registers first because they may not
    # appear in order
    dest = None
    immediate = None # Some instructions may lack it
    vm = None   # Some instructions may lack it
    vscalar = None # Used to mark that we want a scalar shape be generated

    # Generator of operands
    ops_generator = []

    def remove_parentheses(name):
        if name[0] == "(" and name[-1] == ")":
            name = name[1:-1]
        return name

    def operand_index(field):
        result = None
        for (idx, op) in enumerate(i.operand_order):
            op = remove_parentheses(op)
            field_name = str(field)
            if field_name == op or \
                    (field_name.startswith("imm") and op in ["imm", "uimm", "simm"]):
                result = idx
                break
        if result is None:
            raise Exception("Syntax order of operand {} not found".format(field))
        return result

    for f in i.fields:
        if isinstance(f, SlicedImmediate):
            if immediate is None:
                op_idx = operand_index(f)
                immediate = (op_idx, f)
                def capture_ins(i):
                    return lambda : Context.get_next_simm(i.immediate_width())
                ops_generator.append((op_idx, capture_ins(i)))
        elif isinstance(f, Register):
            op_idx = operand_index(f)
            if f.direction == Register.OUT:
                if dest is not None:
                    raise Exception("More than one destination??? {} vs {}".format(dest, f))
                dest = (op_idx, f)
            elif f.direction == Register.IN:
                pass
            else:
                raise Exception("Unhandled direction")
            # Register class
            if f.kind == Register.GPR:
                ops_generator.append((op_idx, lambda : Register.GPR_ABI_NAMES[Context.get_next_gpr()]))
            elif f.kind == Register.FPR:
                ops_generator.append((op_idx, lambda : Register.FPR_ABI_NAMES[Context.get_next_fpr()]))
            elif f.kind == Register.VR:
                def capture_field(f):
                    return lambda : "v{}".format(Context.get_next_vr_modulo(f.scale))
                ops_generator.append((op_idx, capture_field(f)))
            else:
                raise Exception("Unhandled class of register {}".format(f))
        elif isinstance(f, VectorMask):
            op_idx = operand_index(f)
            vm = (op_idx, f)
            ops_generator.append((op_idx, lambda : ""))
        elif isinstance(f, FixedBits):
            pass
        else:
            raise Exception("Unhandled field kind {}".format(f))

    assert(isinstance(i.fields[-1], FixedBits))
    is_store = i.fields[-1].text == "0100111"
    is_load =  i.fields[-1].text == "0000111"
    is_memoperation = is_store or is_load

    # Stores do not have dest but they have vs3
    if dest is None:
        if not is_store :
            raise Exception("Instruction {} is not a store and does "\
                            "not have dest!".format(i.name))
        # FIXME: This is a bit rustic but does the job for now
        found = False
        for f in i.fields:
            if isinstance(f, Register):
                if f.name == "vs3":
                    found = True
                    op_idx = operand_index(f)
                    t = ops_generator.pop(op_idx)
                    ops_generator.insert(0, t)
        if not found:
            raise Exception("'vs3' not found in instruction {}".format(i.name))


    def generate_operands(g):
        res = []
        for i in g:
            res.append((i[0], i[1]()))
        return res

    def render_instruction(instr_name, operands):
        result = instr_name

        # Sort the operands by their index in the assembler
        operands.sort(key = lambda x : x[0])
        # Now remove the order
        operands = map(lambda x : x[1], operands)

        result += " " + operands[0]

        # Sources
        sources = operands[1:]
        for (op_idx, op) in enumerate(sources):
            if op == "":
                continue
            result += ", "
            needs_parenthesis = False
            if op_idx == 0 and is_memoperation:
                needs_parenthesis = True
                result += "("

            result += "{}".format(op)

            if needs_parenthesis:
                result += ")"

        return result

    instr_name = i.name.lower()
    operands = generate_operands(ops_generator)
    result = [ render_instruction(instr_name, operands) ]

    ## Create version with v0.t if there is mask
    if vm is not None:
        mask_true_ops_generator = ops_generator[:]
        for i in range(0, len(mask_true_ops_generator)):
            if mask_true_ops_generator[i][0] == vm[0]:
                mask_true_ops_generator[i] = (vm[0], lambda : "v0.t")
        operands = generate_operands(mask_true_ops_generator)
        result.append(render_instruction(instr_name, operands))

    return  result

def emit_test_parsing(instructions):
    print "# Generated with utils/EPI/process.py"
    print "# RUN: llvm-mc < %s -arch=riscv64 -mattr=+m,+f,+d,+a,+epi | FileCheck %s"
    print ""

    def input_and_check(x):
        return "# CHECK: " + x + "\n" + x

    for i in instructions:
        testcases = generate_instruction_tests(i)
        for tc in testcases:
            print input_and_check(tc)
        print ""

class Generator:
    def __init__(self):
        self.immediate = None
        self.fields = [lambda x : "|"]
        pass

    def copy(self):
        r = Generator()
        r.immediate = self.immediate
        r.fields = self.fields[:]
        return r

def generate_expected_encodings(i):
    vm = -1   # Some instructions may lack it
    dest = None
    generator = Generator()
    for f in i.fields:
        if isinstance(f, VectorMask):
            vm = len(generator.fields)
            generator.fields.append(lambda x : "1|")
        elif isinstance(f, Register):
            if f.direction == Register.OUT:
                if dest is not None:
                    raise Exception("More than one destination??? {} vs {}".format(dest, f))
                dest = f
            elif f.direction == Register.IN:
                pass
            else:
                raise Exception("Unhandled direction")

            if f.kind == Register.GPR:
                generator.fields.append(lambda x : "{:05b}|".format(Context.get_next_gpr()))
            elif f.kind == Register.FPR:
                generator.fields.append(lambda x : "{:05b}|".format(Context.get_next_fpr()))
            elif f.kind == Register.VR:
                def capture_field(f):
                    return lambda x : "{:05b}|".format(Context.get_next_vr_modulo(f.scale))
                generator.fields.append(capture_field(f))
            else:
                raise Exception("unhandled register kind {}".format(f))
        elif isinstance(f, FixedBits):
            def capture(t):
                return lambda x : t
            generator.fields.append(capture(f.text + "|"))
            pass
        elif isinstance(f, SlicedImmediate):
            if generator.immediate is None:
                def capture(i):
                    return lambda : Context.get_next_simm_as_imm(i)
                generator.immediate = capture(i.immediate_width())
            def select_bits(u, l):
                def select_bits_impl(i):
                    s = ("{:016b}".format(i))[::-1]
                    return s[l:u + 1][::-1] + "|"
                return select_bits_impl
            generator.fields.append(select_bits(f.upper(), f.lower()))
        else:
            raise Exception("unhandled field {}".format(f))


    def execute_generator(g):
        i = None
        if g.immediate is not None:
            i = (g.immediate)()
        s = ""
        for l in g.fields:
            s += l(i)
        # assert(len(s) == 32)
        return s

    bits = execute_generator(generator)
    result = [bits]

    if vm >= 0:
        mask_true_generator = generator.copy()
        mask_true_generator.fields = generator.fields[:]
        mask_true_generator.fields[vm] = lambda x : "0|";
        result.append(execute_generator(mask_true_generator))

    return result

def emit_test_encoding(instructions):
    print "# Generated witu utils/EPI/process.py"
    print "# RUN: llvm-mc < %s -arch=riscv64 -mattr=+m,+f,+d,+a,+epi -show-encoding | FileCheck %s"
    print ""
    testcases_list = []
    encodings_list = []
    for i in instructions:
        testcases_list.append(generate_instruction_tests(i))

    Context.reset() # Synchronize both generations

    for i in instructions:
        encodings_list.append(generate_expected_encodings(i))

    for (testcases, encodings) in zip(testcases_list, encodings_list):
        if len(testcases) != len(encodings):
            print testcases
            print encodings
            raise Exception("Number of parsing testcases {} and encoding " \
                    "testcases {} does not match".format(len(testcases), len(encodings)))
        assert(len(testcases) == len(encodings))
        for (t, e_orig) in zip(testcases, encodings):
            e = e_orig.replace("|", "")
            assert(len(e) == 32)
            hex_values = []
            for i in range(0, 4):
                hex_values.append("0x{:02x}".format(int(e[8*i:8*i + 8], 2)))
            hex_values = hex_values[::-1]
            hex_values_str = "[{}]".format(",".join(hex_values))
            print "# Encoding: {}".format(e_orig)
            print "# CHECK: {}".format(t)
            print "# CHECK-SAME: {}".format(hex_values_str)
            print "{}".format(t)
        print ""

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser(description="Process instructions")
    args.add_argument("input_filename",
                      help="File with the instruction description")
    args.add_argument("operation",
                      default="print",
                      nargs='?',
                      choices=["print", "test_parsing",
                               "test_encoding"],
                      help="What to do with the instructions")
    args = args.parse_args()

    iparser = InstructionParser()
    iparser.parse_input(args.input_filename)

    if args.operation == "print":
        for instruction in iparser.instructions:
            print instruction
    elif args.operation == "test_parsing":
        emit_test_parsing(iparser.instructions)
    elif args.operation == "test_encoding":
        emit_test_encoding(iparser.instructions)
    else:
        raise Exception("Unhandled action {}".format(args.operation))
