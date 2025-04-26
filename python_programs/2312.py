from z3 import *

artists_sort, (A, B, C, D) = EnumSort('artists', ['A', 'B', 'C', 'D'])
professions_sort, (dancer, painter, singer, writer) = EnumSort('professions', ['dancer', 'painter', 'singer', 'writer'])
artists = [A, B, C, D]
professions = [dancer, painter, singer, writer]
role_of = Function('role_of', artists_sort, professions_sort)

pre_conditions = []
pre_conditions.append(Distinct([role_of(a) for a in artists]))
pre_conditions.append(And(role_of(A) != singer, role_of(C) != singer))
pre_conditions.append(And(role_of(A) != painter, role_of(C) != painter))
pre_conditions.append(role_of(A) == writer)
pre_conditions.append(role_of(B) == singer)
pre_conditions.append(A != C)
pre_conditions.append(Distinct([role_of(a) for a in artists]))
pre_conditions.append(And(role_of(A) != singer, role_of(C) != singer))
pre_conditions.append(And(role_of(A) != painter, role_of(C) != painter))
pre_conditions.append(role_of(A) == writer)
pre_conditions.append(role_of(B) == singer)
pre_conditions.append(A != C)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(And(role_of(A) != singer, role_of(C) != singer), True)
verification_results.append(result_0)
result_1 = is_deduced(And(role_of(A) == writer, role_of(B) == singer, And(role_of(A) != painter, role_of(C) != painter)), role_of(D) == painter)
verification_results.append(result_1)
result_2 = is_deduced(True, role_of(A) == writer)
verification_results.append(result_2)
result_3 = is_deduced(True, role_of(B) == singer)
verification_results.append(result_3)
result_4 = is_deduced(True, A != C)
verification_results.append(result_4)
result_5 = is_deduced(True, role_of(C) == painter)
verification_results.append(result_5)

# Print all verification results
print('All verification results:', verification_results)
