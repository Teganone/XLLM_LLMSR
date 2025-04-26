from z3 import *

candidates_sort, (F, G, H, I, W, X, Y) = EnumSort('candidates', ['F', 'G', 'H', 'I', 'W', 'X', 'Y'])
departments_sort, (public_relations, production, sales) = EnumSort('departments', ['public_relations', 'production', 'sales'])
candidates = [F, G, H, I, W, X, Y]
departments = [public_relations, production, sales]
dept_of = Function('dept_of', candidates_sort, departments_sort)

pre_conditions = []
pre_conditions.append(Sum([dept_of(c) == public_relations for c in candidates]) == 1)
pre_conditions.append(Sum([dept_of(c) == production for c in candidates]) == 3)
pre_conditions.append(Sum([dept_of(c) == sales for c in candidates]) == 3)
pre_conditions.append(dept_of(H) == dept_of(Y))
pre_conditions.append(dept_of(F) != dept_of(G))
pre_conditions.append(Implies(dept_of(X) == sales, dept_of(W) == production))
pre_conditions.append(dept_of(F) == production)
pre_conditions.append(Sum([dept_of(c) == public_relations for c in candidates]) == 1)
pre_conditions.append(Sum([dept_of(c) == production for c in candidates]) == 3)
pre_conditions.append(Sum([dept_of(c) == sales for c in candidates]) == 3)
pre_conditions.append(dept_of(H) == dept_of(Y))
pre_conditions.append(dept_of(F) != dept_of(G))
pre_conditions.append(Implies(dept_of(X) == sales, dept_of(W) == production))
pre_conditions.append(dept_of(F) == production)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(True, dept_of(F) == production)
verification_results.append(result_0)
result_1 = is_deduced(True, dept_of(G) != dept_of(F))
verification_results.append(result_1)
result_2 = is_deduced(dept_of(F) == production, Or(dept_of(G) == public_relations, dept_of(G) == sales))
verification_results.append(result_2)
result_3 = is_deduced(True, dept_of(X) == sales)
verification_results.append(result_3)
result_4 = is_deduced(dept_of(X) == sales, dept_of(W) == production)
verification_results.append(result_4)
result_5 = is_deduced(True, And(dept_of(H) == public_relations, dept_of(Y) == public_relations))
verification_results.append(result_5)

# Print all verification results
print('All verification results:', verification_results)
