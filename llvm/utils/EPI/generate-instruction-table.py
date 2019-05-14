#!/usr/bin/env python2

import string
import re
import collections


class Field(object):
    def __init__(self, size):
        self.size = size

class FieldAlternatives(Field):
    def __init__(self, options):
        assert(len(options) > 1)
        size = options[0].size
        for f in options[1:]:
            if size != f.size:
                raise Exception("Alternative field '{}' has size {} which is different to the size {} of field '{}'" \
                        .format(str(f), f.size, size, str(options[0])))
        super(FieldAlternatives, self).__init__(size)
        self.options = options

    def __str__(self):
        t = map(lambda x : str(x), self.options)
        return "/".join(t)

    def matches_name(self, name):
        return any(map(lambda x : x.matches_name(name), self.options))

class VectorMask(Field):
    def __init__(self):
        super(VectorMask, self).__init__(1)
        pass

    def __str__(self):
        return "vm"

    def matches_name(self, name):
        return name == "vm"

class Register(Field):
    def __init__(self, name):
        super(Register, self).__init__(5)
        self.name = name

    def __str__(self):
        return self.name

    def matches_name(self, name):
        return self.name == name

class IndirectRegister(Register):
    def __init__(self, name):
        super(Register, self).__init__()
        self.name = IndirectRegister.remove_parentheses(name)

    def __str__(self):
        return "(" + self.name + ")"

    def matches_name(self, name):
        return self.name == name

    @staticmethod
    def remove_parentheses(name):
        assert(name[0] == "(")
        assert(name[-1] == ")")
        return name[1:-1]

class Immediate(Field):
    def __init__(self, name, size):
        super(Immediate, self).__init__(size)
        self.name = name

    def __str__(self):
        return self.name

    def matches_name(self, name):
        return name in ["imm", "simm", "uimm"]

class FixedBits(Field):
    def __init__(self, bits):
        super(FixedBits, self).__init__(len(bits))
        self.bits = bits

    def __str__(self):
        return self.bits

    def matches_name(self, name):
        return False

class Format(object):
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields

    def __str__(self):
        t = map(lambda x : str(x), self.fields)
        return "{}: {}".format(self.name, "|".join(t))

class Constant(object):
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def __str__(self):
        return self.name

    def matches_name(self, name):
        return self.name == name

class InstructionSyntax(object):
    def __init__(self, name, operands):
        self.name = name
        self.operands = operands

    def __str__(self):
        return "{} {}".format(self.name, ", ".join(self.operands))

class Encoding(object):
    def __init__(self, instruction, format_spec, assignments, defined_fields):
        self.instruction = instruction
        self.format = format_spec
        self.assignments = assignments
        self.defined_fields = defined_fields

    def __str__(self):
        t = []
        for (n, v) in self.assignments.iteritems():
            t.append("{}={}".format(n, v))

        return "{}|{}|{}".format(self.instruction.name, self.format.name, "|".join(t))

class Assignment(object):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

class InstrDesc:
    FORMAT_HEADER = "--- FORMAT"
    ASSEMBLER_HEADER = "--- ASSEMBLER"
    ENCODING_HEADER = "--- ENCODING"

    GPREG = re.compile("^(rs[0-9]+)|(rd)$")
    INDIRECT_GPREG = re.compile("^\((rs[0-9]+)|(rd)\)$")
    VREG = re.compile("^(vs[0-9]+)|(vd)$")
    IMM = re.compile("^[us]?imm:[0-9]+$")
    VMASK = re.compile("^vm$")
    BITS = re.compile("^[01]+$")

    ENCODING_ASSIGNMENT = re.compile("^[0-9a-z.]+=[01]+$")

    # SECTION
    SECTION_PRELUDE = 0
    SECTION_FORMAT = 1
    SECTION_ASSEMBLER = 2
    SECTION_ENCODING = 3

    def __init__(self, input_filename):
        self.input_filename = input_filename
        self.formats = collections.OrderedDict()
        self.instructions = collections.OrderedDict()
        self.encodings = collections.OrderedDict()

    def error_at(self, linenum, error_message):
        raise Exception("{}:{}: {}".format(self.input_filename, \
                                           linenum, \
                                           error_message))

    def parse_input(self):
        input_file = open(self.input_filename, "r")
        self.current_section = InstrDesc.SECTION_PRELUDE

        lastline = 1
        for (linenum, line) in enumerate(input_file):
            linenum += 1
            line = line.strip()
            # Skip comments and empty lines
            if line == "" or line[0] == "#":
                continue
            # This can be done in a nicer way
            if self.current_section == InstrDesc.SECTION_FORMAT:
                self.process_format_line(linenum, line)
            elif self.current_section == InstrDesc.SECTION_ASSEMBLER:
                self.process_assembler_line(linenum, line)
            elif self.current_section == InstrDesc.SECTION_ENCODING:
                self.process_encoding_line(linenum, line)
            elif self.current_section == InstrDesc.SECTION_PRELUDE:
                self.process_prelude_line(linenum, line)
            else:
                raise Exception("Code unreachable")
            lastline = linenum

        if self.current_section != InstrDesc.SECTION_ENCODING:
            self.error_at(lastline, "the input file does not contain all the sections")

        # Check all the instructions have an encoding
        for instr_name in self.instructions:
            if instr_name not in self.encodings:
                # FIXME: We could use the line of the instruction
                self.error_at(lastline, "Instruction '{}' does not have any defined encoding".format(instr_name))

    def process_prelude_line(self, linenum, line):
        if line != InstrDesc.FORMAT_HEADER:
            self.error_at(linenum,
                "the input must start with a '{}' line (line is '{}')".format(InstrDesc.FORMAT_HEADER, line))
        else:
            self.current_section = InstrDesc.SECTION_FORMAT

    def process_format_line(self, linenum, line):
        if line == InstrDesc.ASSEMBLER_HEADER:
            self.current_section = InstrDesc.SECTION_ASSEMBLER
        else:
            fields = line.split("|")
            format_name = fields.pop()
            processed_fields = []
            for f in fields:
                options = f.split("/")
                processed_options = []
                for o in options:
                    if InstrDesc.VMASK.match(o):
                        processed_options.append(VectorMask())
                    elif InstrDesc.GPREG.match(o) or InstrDesc.VREG.match(o):
                        processed_options.append(Register(o))
                    elif InstrDesc.INDIRECT_GPREG.match(o):
                        processed_options.append(IndirectRegister(o))
                    elif InstrDesc.IMM.match(o):
                        (name, size) = o.split(":")
                        processed_options.append(Immediate(name, int(size)))
                    elif InstrDesc.BITS.match(o):
                        processed_options.append(FixedBits(o))
                    else: # Assume this is a constant field
                        if ':' not in o:
                            raise Exception("Field {} needs a size".format(o));
                        (name, size) = o.split(":")
                        processed_options.append(Constant(name, int(size)))

                if len(processed_options) == 1:
                    processed_fields.append(processed_options[0])
                else:
                    processed_fields.append(FieldAlternatives(processed_options))

            self.formats[format_name] = Format(format_name, processed_fields)

    def process_assembler_line(self, linenum, line):
        if line == InstrDesc.ENCODING_HEADER:
            self.current_section = InstrDesc.SECTION_ENCODING
        else:
            # This is not very elegant or robust
            components = line.split(" ")
            instruction_name = components.pop(0)
            operands = []

            for c in components:
                if c[-1] == ",":
                    c = c[:-1]
                operands.append(c)

            self.instructions[instruction_name] = InstructionSyntax(instruction_name, operands)

    def process_encoding_line(self, linenum, line):
        fields = line.split("|")
        instruction_name = fields.pop(0)
        if instruction_name not in self.instructions:
            self.error_at(linenum, "instruction '{}' is unknown".format(instruction_name))
        instr = self.instructions[instruction_name]

        format_name = fields.pop(0)
        if format_name not in self.formats:
            self.error_at(linenum, "encoding format '{}' is unknown".format(format_name))

        format_spec = self.formats[format_name]

        defined_fields = [None] * len(format_spec.fields)
        # Mark fixed fields as defined
        for (i, field) in enumerate(format_spec.fields):
            if isinstance(field, FixedBits):
                defined_fields[i] = (field, field)

        # Mark fields passed as operands as defined
        for op in instr.operands:
            used = False
            for (i, field) in enumerate(format_spec.fields):
                # FIXME:
                if InstrDesc.INDIRECT_GPREG.match(op):
                    op = IndirectRegister.remove_parentheses(op)
                if field.matches_name(op):
                    if defined_fields[i] is not None:
                        self.error_at("operand '{}' defines field '{}' but it has already been defined by '{}'".format(op, field, define_fields[i][0]))
                    defined_fields[i] = (op, field)
                    used = True
            if not used:
                self.error_at(linenum, "operand '{}' does not participate in the encoding of the instruction '{}'".format(op, instruction_name))

        assignments = collections.OrderedDict()
        for assign in fields:
            if not InstrDesc.ENCODING_ASSIGNMENT.match(assign):
                self.error_at(linenum, "element '{}' does not have the form of a field assign <FIELD>=<bits>")
            (assign_name, assign_value) = assign.split("=")
            if not InstrDesc.BITS.match(assign_value):
                self.error_at(linenum, "assigned value '{}' to field '{}' is not a binary constant".format(assign_value, field))
            for (i, field) in enumerate(format_spec.fields):
                if field.matches_name(assign_name):
                    if len(assign_value) != field.size:
                        self.error_at(linenum, "assigned value '{}' to field '{}' is not of size {}".format(assign_value, field, field.size))
                    if defined_fields[i] is not None:
                        self.error_at(linenum, "assigned value '{}' to field '{}' that already has value '{}'".format(assign_value, field, define_fields[i][0]))
                    defined_fields[i] = (Assignment(assign_value), field)
            assignments[assign_name] = assign_value

        # Now check all the fields are defined
        for (i, field) in enumerate(format_spec.fields):
            if defined_fields[i] is None:
                self.error_at(linenum, "field '{}' of format '{}' has not been defined for instruction '{}'".format(str(field), format_name, instruction_name))

        self.encodings[instruction_name] = Encoding(instr, format_spec, assignments, defined_fields)

    def dump(self):
        print "{}".format(InstrDesc.FORMAT_HEADER)
        for fmt in self.formats.itervalues():
            print fmt
        print "{}".format(InstrDesc.ASSEMBLER_HEADER)
        for asm in self.instructions.itervalues():
            print asm
        print "{}".format(InstrDesc.ENCODING_HEADER)
        for enc in self.encodings.itervalues():
            print enc

    def emit_listing_table(self):
        for encoding in self.encodings.itervalues():
            field_render = []
            # Instruction name
            field_render.append(encoding.instruction.name.upper())
            # Used to retrieve the operand order
            instr_syntax = ",".join(encoding.instruction.operands)
            field_render.append(instr_syntax)

            for (value, field) in encoding.defined_fields:
                if isinstance(field, Immediate):
                    field_render.append("{}[{}:0]".format(value, field.size - 1))
                else:
                    field_render.append(str(value))

            print "|".join(field_render)


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser(description="Process instructions")
    args.add_argument("input_filename",
                      help="File with the instruction description")
    args.add_argument("operation",
                      default="print",
                      nargs='?',
                      choices=["print", "generate_table"],
                      help="What to do with the instructions")
    args = args.parse_args()

    iparser = InstrDesc(args.input_filename)
    iparser.parse_input()
    if args.operation == "print":
        iparser.dump()
    elif args.operation == "generate_table":
        iparser.emit_listing_table()
    else:
        raise Exception("Unhandled action {}".format(args.operation))
