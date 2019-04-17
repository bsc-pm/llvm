#!/usr/bin/env python2

from preprocess_builtins import *
from builtin_parser import *
from type_render import *

def emit_declaration(builtin_name, prototype):
    (return_type, argument_types) = parse_type(prototype)
    return_type_str = TypeRender(return_type).render()
    argument_types_str = map(lambda x : TypeRender(x).render(), argument_types)
    print "{} __builtin_epi_{}({});".format(return_type_str, \
            builtin_name, ", ".join(argument_types_str))

if __name__ == "__main__":
    for (builtin_name, prototype) in preprocess_builtins():
        emit_declaration(builtin_name, prototype)
