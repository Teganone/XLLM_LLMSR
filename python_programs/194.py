from z3 import *

islands_sort, (E, F, G, H, I) = EnumSort('islands', ['E', 'F', 'G', 'H', 'I'])
islands = [E, F, G, H, I]
position_of = Function('position_of', islands_sort, IntSort())

pre_conditions = []
pre_conditions.append(Distinct([position_of(p) for p in islands]))
pre_conditions.append(position_of(H) == position_of(F) + 1)
pre_conditions.append(Or(position_of(I) == position_of(E) + 1, position_of(E) == position_of(I) + 1))
pre_conditions.append(position_of(G) < position_of(F))
pre_conditions.append(Distinct([position_of(p) for p in islands]))
pre_conditions.append(position_of(H) == position_of(F) + 1)
pre_conditions.append(Or(position_of(I) == position_of(E) + 1, position_of(E) == position_of(I) + 1))
pre_conditions.append(position_of(G) < position_of(F))

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(position_of(H) == position_of(F) + 1, position_of(F) < position_of(H))
verification_results.append(result_0)
result_1 = is_deduced(Or(position_of(I) == position_of(E) + 1, position_of(E) == position_of(I) + 1), True)
verification_results.append(result_1)
result_2 = is_deduced(position_of(G) < position_of(F), True)
verification_results.append(result_2)

# Print all verification results
print('All verification results:', verification_results)
