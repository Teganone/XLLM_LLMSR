from collections import OrderedDict
from src.z3_solver.code_translator import *
import subprocess
from subprocess import check_output
from os.path import join
import os

class LLMSR_Z3_Program:
    def __init__(self, logic_program:str) -> None:
        self.logic_program = logic_program
        try:
            self.parse_logic_program_verification()
            self.standard_code = self.to_standard_code()
        except Exception as e:
            print(f"Error during initialization: {e}")
            self.standard_code = None
            self.flag = False
            return
        
        self.flag = True

        # create the folder to save the Pyke program
        cache_dir = os.path.join(os.path.dirname(__file__), '.cache_program')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir


    def parse_logic_program_verification(self):
        # split the logic program into different parts
        lines = [x for x in self.logic_program.splitlines() if not x.strip() == ""]
        decleration_start_index = lines.index("# Declarations")
        constraint_start_index = lines.index("# Constraints")
        verification_start_index = lines.index("# Verifications")

        declaration_statements = lines[decleration_start_index + 1:constraint_start_index]
        constraint_statements = lines[constraint_start_index + 1:verification_start_index]
        verification_statements = lines[verification_start_index + 1:]
        try:
            (self.declared_enum_sorts, self.declared_int_sorts, self.declared_lists, self.declared_functions, self.variable_constrants) = self.parse_declaration_statements(declaration_statements)
            # print(self.declared_enum_sorts, '\n', self.declared_int_sorts, '\n', self.declared_lists, '\n', self.declared_functions, '\n', self.variable_constrants)
            self.constraints, self.constraints_text = [x.split(':::')[0].strip() for x in constraint_statements], [x.split(':::')[1].strip() for x in constraint_statements]
            self.verifications, self.verifications_text = [x.split(':::')[0].strip() for x in verification_statements], [x.split(':::')[1].strip() for x in verification_statements]
            
        except Exception as e:
            return False
        
        return True

    def __repr__(self):
        return f"LSATSatProblem:\n\tDeclared Enum Sorts: {self.declared_enum_sorts}\n\tDeclared Lists: {self.declared_lists}\n\tDeclared Functions: {self.declared_functions}\n\tConstraints: {self.constraints}\n\tVerifications: {self.verifications}"

    def parse_declaration_statements(self, declaration_statements):
        enum_sort_declarations = OrderedDict()
        int_sort_declarations = OrderedDict()
        function_declarations = OrderedDict()

        preprocessed_declaration_statements = []
        for s in declaration_statements:
            if ":::" in s:
                s = s.split(":::")[0].strip()
            preprocessed_declaration_statements.append(s)
    
        pure_declaration_statements = [x for x in preprocessed_declaration_statements if "Sort" in x or "Function" in x]
        variable_constrant_statements = [x for x in preprocessed_declaration_statements if not "Sort" in x and not "Function" in x]
        for s in pure_declaration_statements:
            if "EnumSort" in s:
                sort_name = s.split("=")[0].strip()
                sort_member_str = s.split("=")[1].strip()[len("EnumSort("):-1]
                sort_members = [x.strip() for x in sort_member_str[1:-1].split(",")]
                enum_sort_declarations[sort_name] = sort_members
            elif "IntSort" in s:
                sort_name = s.split("=")[0].strip()
                sort_member_str = s.split("=")[1].strip()[len("IntSort("):-1]
                sort_members = [x.strip() for x in sort_member_str[1:-1].split(",")]
                int_sort_declarations[sort_name] = sort_members
            elif "Function" in s:
                function_name = s.split("=")[0].strip()
                if "->" in s and "[" not in s:
                    function_args_str = s.split("=")[1].strip()[len("Function("):]
                    function_args_str = function_args_str.replace("->", ",").replace("(", "").replace(")", "")
                    function_args = [x.strip() for x in function_args_str.split(",")]
                    function_declarations[function_name] = function_args
                elif "->" in s and "[" in s:
                    function_args_str = s.split("=")[1].strip()[len("Function("):-1]
                    function_args_str = function_args_str.replace("->", ",").replace("[", "").replace("]", "")
                    function_args = [x.strip() for x in function_args_str.split(",")]
                    function_declarations[function_name] = function_args
                else:
                    # legacy way
                    function_args_str = s.split("=")[1].strip()[len("Function("):-1]
                    function_args = [x.strip() for x in function_args_str.split(",")]
                    function_declarations[function_name] = function_args
            else:
                raise RuntimeError("Unknown declaration statement: {}".format(s))

        declared_enum_sorts = OrderedDict()
        declared_lists = OrderedDict()
        self.declared_int_lists = OrderedDict()

        declared_functions = function_declarations
        already_declared = set()
        for name, members in enum_sort_declarations.items():
            # all contained by other enum sorts
            if all([x not in already_declared for x in members]):
                declared_enum_sorts[name] = members
                already_declared.update(members)
            declared_lists[name] = members

        for name, members in int_sort_declarations.items():
            self.declared_int_lists[name] = members
            # declared_lists[name] = members

        return declared_enum_sorts, int_sort_declarations, declared_lists, declared_functions, variable_constrant_statements
    
    def to_standard_code(self):
        declaration_lines = []
        # translate enum sorts
        for name, members in self.declared_enum_sorts.items():
            declaration_lines += CodeTranslator.translate_enum_sort_declaration(name, members)

        # translate int sorts
        for name, members in self.declared_int_sorts.items():
            declaration_lines += CodeTranslator.translate_int_sort_declaration(name, members)
        # translate lists
        for name, members in self.declared_lists.items():
            declaration_lines += CodeTranslator.translate_list_declaration(name, members)
        scoped_list_to_type = {}
        for name, members in self.declared_lists.items():
            if all(x.isdigit() for x in members):
                scoped_list_to_type[name] = CodeTranslator.ListValType.INT
            else:
                scoped_list_to_type[name] = CodeTranslator.ListValType.ENUM

        for name, members in self.declared_int_lists.items():
            scoped_list_to_type[name] = CodeTranslator.ListValType.INT
        
        # translate functions
        for name, args in self.declared_functions.items():
            declaration_lines += CodeTranslator.translate_function_declaration(name, args)

        pre_condidtion_lines = []

        for constraint in self.constraints:
            pre_condidtion_lines += CodeTranslator.translate_constraint(constraint, scoped_list_to_type)

        # additional function scope control
        for name, args in self.declared_functions.items():
            if args[-1] in scoped_list_to_type and scoped_list_to_type[args[-1]] == CodeTranslator.ListValType.INT:
                # FIX
                if args[-1] in self.declared_int_lists:
                    continue
                
                list_range = [int(x) for x in self.declared_lists[args[-1]]]
                assert list_range[-1] - list_range[0] == len(list_range) - 1
                scoped_vars = [x[0] + str(i) for i, x in enumerate(args[:-1])]
                func_call = f"{name}({', '.join(scoped_vars)})"

                additional_cons = "ForAll([{}], And({} <= {}, {} <= {}))".format(
                    ", ".join([f"{a}:{b}" for a, b in zip(scoped_vars, args[:-1])]),
                    list_range[0], func_call, func_call, list_range[-1]
                )
                pre_condidtion_lines += CodeTranslator.translate_constraint(additional_cons, scoped_list_to_type)

        for constraint in self.constraints:
            pre_condidtion_lines += CodeTranslator.translate_constraint(constraint, scoped_list_to_type)

        # Process verification blocks
        verification_blocks = [(CodeTranslator.translate_constraint(verification, scoped_list_to_type), index) for verification, index in zip(self.verifications, self.verifications_text)]
        
        return CodeTranslator.assemble_standard_code(declaration_lines, pre_condidtion_lines, option_blocks=None, verification_blocks=verification_blocks)
    
    def execute_program(self):
        filename = join(self.cache_dir, f'tmp.py')
        with open(filename, "w") as f:
            f.write(self.standard_code)
        try:
            output = check_output(["python", filename], stderr=subprocess.STDOUT, timeout=1.0)
        except subprocess.CalledProcessError as e:
            outputs = e.output.decode("utf-8").strip().splitlines()[-1]
            return None, outputs
        except subprocess.TimeoutExpired:
            return None, 'TimeoutError'
        output = output.decode("utf-8").strip()
        result = output.splitlines()
        if len(result) == 0:
            return None, 'No Output'
        
        return result, ""
    
    def answer_mapping(self, answer):
        mapping = {'(A)': 'A', '(B)': 'B', '(C)': 'C', '(D)': 'D', '(E)': 'E',
                   'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E'}
        return mapping[answer[0].strip()]

if __name__=="__main__":
    logic_program = '''
# Declarations
students = EnumSort([G, H, L, M, U, W, Z])
countries = EnumSort([UK, US])
goes_to = Function([students] -> [countries])

# Constraints
Implies(goes_to(G) == UK, goes_to(H) == US) ::: (1) If G goes to the UK, then H To the United States
Implies(goes_to(L) == UK, And(goes_to(M) == US, goes_to(U) == US)) ::: (2) If L goes to the UK, both M and U go to the US
goes_to(W) != goes_to(Z) ::: (3) The country W went to was different from the country Z went to
goes_to(U) != goes_to(G) ::: (4) The country where U goes is different from the country where G goes
Implies(goes_to(Z) == UK, goes_to(H) == UK) ::: (5) If Z goes to the UK, then H also goes to the UK
goes_to(G) == US ::: If G goes to the United States

# Verifications
is_deduced(goes_to(G) == US, goes_to(G) != UK) ::: (1) Condition (1) is not applicable
is_deduced(goes_to(G) == US, True) ::: (2) Condition (2) is also not applicable
is_deduced(goes_to(W) != goes_to(Z), False) ::: (3) Condition (3) does not provide any information about H, M, U, or W
is_deduced(And(goes_to(U) != goes_to(G), goes_to(G) == US), goes_to(U) == UK) ::: (4) U must go to the UK
is_deduced(goes_to(G) == US, True) ::: (5) Condition (5) is not applicable
'''

    z3_program = LLMSR_Z3_Program(logic_program)
    print(z3_program.standard_code)

    output, error_message = z3_program.execute_program()
    print(output)
