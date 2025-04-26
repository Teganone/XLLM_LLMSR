from z3 import *

islands_sort, (E, F, G, H, I) = EnumSort('islands', ['E', 'F', 'G', 'H', 'I'])
islands = [E, F, G, H, I]
position_of = Function('position_of', islands_sort, IntSort())

pre_conditions = []
pre_conditions.append(Distinct([position_of(i) for i in islands]))
i = Const('i', islands_sort)
pre_conditions.append(ForAll([i], And(1 <= position_of(i), position_of(i) <= 5)))
pre_conditions.append(position_of(F) + 1 == position_of(H))
pre_conditions.append(Or(position_of(I) == position_of(E) + 1, position_of(E) == position_of(I) + 1))
pre_conditions.append(position_of(G) < position_of(F))
pre_conditions.append(Implies(position_of(I) < position_of(G), And(position_of(I) == 1, position_of(E) == 2, position_of(G) == 3, position_of(F) == 4, position_of(H) == 5)))
pre_conditions.append(Distinct([position_of(i) for i in islands]))
i = Const('i', islands_sort)
pre_conditions.append(ForAll([i], And(1 <= position_of(i), position_of(i) <= 5)))
pre_conditions.append(position_of(F) + 1 == position_of(H))
pre_conditions.append(Or(position_of(I) == position_of(E) + 1, position_of(E) == position_of(I) + 1))
pre_conditions.append(position_of(G) < position_of(F))
pre_conditions.append(Implies(position_of(I) < position_of(G), And(position_of(I) == 1, position_of(E) == 2, position_of(G) == 3, position_of(F) == 4, position_of(H) == 5)))

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(position_of(F) + 1 == position_of(H), True)
verification_results.append(result_0)
result_1 = is_deduced(Or(position_of(I) == position_of(E) + 1, position_of(E) == position_of(I) + 1), True)
verification_results.append(result_1)
result_2 = is_deduced(position_of(G) < position_of(F), True)
verification_results.append(result_2)
result_3 = is_deduced(Implies(position_of(I) < position_of(G), position_of(I) == 1), True)
verification_results.append(result_3)
result_4 = is_deduced(Implies(position_of(I) < position_of(G), position_of(E) == 2), True)
verification_results.append(result_4)
result_5 = is_deduced(Implies(position_of(I) < position_of(G), position_of(G) == 3), True)
verification_results.append(result_5)
result_6 = is_deduced(Implies(position_of(I) < position_of(G), And(position_of(F) == 4, position_of(H) == 5)), True)
verification_results.append(result_6)

# Print all verification results
print('All verification results:', verification_results)
