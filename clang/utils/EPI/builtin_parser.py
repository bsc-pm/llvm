from type_builder import *

def parse_type(prototype):
    class State:
        i = 0
        end = len(prototype)

    class Helper:
        digits = map(lambda x : str(x), range(0, 10))

    def peek():
        if State.i >= State.end:
            return None
        return prototype[State.i]

    def skip():
        if State.i < State.end:
            State.i = State.i + 1
        return peek()

    def parse_single_type():
        # print "Input '{}'".format(prototype[State.i:])
        parse_start = State.i
        c = peek()
        built_type = TypeBuilder()
        # Can be 1 (prefix), 2 (main), o 3 (suffix), 4 (unhandled so far)
        part_of_type = 1
        basic_types =['b', 'c', 's', 'i', 'f', 'd', 'v']
        done = False
        while c is not None:
            # print "->{}<-".format(c)
            # Prefixes
            if part_of_type == 1:
                if c == 'L':
                    built_type.set_long()
                elif c == 'W': # 64-bit!
                    built_type.set_long()
                elif c == 'U':
                    built_type.set_unsigned()
                elif c == 'S':
                    built_type.set_signed()
                elif c == 'Q':
                    built_type.set_scalable_vector()
                elif c == 'V':
                    built_type.set_vector()
                    c = skip()
                    l = 0
                    while c in Helper.digits:
                        l = 10*l + int(c)
                        c = skip()
                    built_type.set_vector_length(l)
                    continue
                elif c == 'I':
                    built_type.set_must_be_constant()
                else:
                    part_of_type = 2
                    continue
            elif part_of_type == 2:
                part_of_type = 3
                if c in basic_types and built_type.has_basic_type():
                    done = True
                    break;
                if c == 'b':
                    built_type.set_basic_type(TypeBuilder.BOOL)
                elif c == 'c':
                    built_type.set_basic_type(TypeBuilder.CHAR)
                elif c == 's':
                    built_type.set_short()
                    built_type.set_basic_type(TypeBuilder.INT)
                elif c == 'i':
                    built_type.set_basic_type(TypeBuilder.INT)
                elif c == 'f':
                    built_type.set_basic_type(TypeBuilder.FLOAT)
                elif c == 'd':
                    built_type.set_basic_type(TypeBuilder.DOUBLE)
                elif c == 'v':
                    built_type.set_basic_type(TypeBuilder.VOID)
                else:
                    continue
            elif part_of_type == 3:
                if c == '*':
                    built_type.set_pointer()
                else:
                    if built_type.has_basic_type():
                        done = True
                    break
            # Reach here only if we handled the letter
            c = skip()

        if c is not None and not done:
            raise Exception("Unhandled '{}'".format(c))

        parse_end = State.i
        # print "Parsed '{}'".format(prototype[parse_start:parse_end])
        return built_type

    return_type = parse_single_type()
    arguments = []
    while peek() is not None:
        arguments.append(parse_single_type())

    return (return_type, arguments)
