from z3 import *

singers_sort, (F, G, L, K, H, M) = EnumSort('singers', ['F', 'G', 'L', 'K', 'H', 'M'])
accompaniments_sort, (X, Y, W) = EnumSort('accompaniments', ['X', 'Y', 'W'])
singers = [F, G, L, K, H, M]
accompaniments = [X, Y, W]
accompany = Function('accompany', singers_sort, accompaniments_sort)

pre_conditions = []
pre_conditions.append(Sum([accompany(s) == X for s in singers]) == 2)
pre_conditions.append(Sum([accompany(s) == Y for s in singers]) == 2)
pre_conditions.append(Sum([accompany(s) == W for s in singers]) == 2)
pre_conditions.append(Implies(accompany(F) == X, accompany(L) == W))
pre_conditions.append(Implies(accompany(G) != X, accompany(M) == Y))
pre_conditions.append(Or(accompany(H) == X, accompany(H) == Y))
pre_conditions.append(And(accompany(F) != accompany(G), accompany(L) != accompany(K), accompany(H) != accompany(M)))
pre_conditions.append(accompany(L) == X)
pre_conditions.append(accompany(H) == X)
pre_conditions.append(Sum([accompany(s) == X for s in singers]) == 2)
pre_conditions.append(Sum([accompany(s) == Y for s in singers]) == 2)
pre_conditions.append(Sum([accompany(s) == W for s in singers]) == 2)
pre_conditions.append(Implies(accompany(F) == X, accompany(L) == W))
pre_conditions.append(Implies(accompany(G) != X, accompany(M) == Y))
pre_conditions.append(Or(accompany(H) == X, accompany(H) == Y))
pre_conditions.append(And(accompany(F) != accompany(G), accompany(L) != accompany(K), accompany(H) != accompany(M)))
pre_conditions.append(accompany(L) == X)
pre_conditions.append(accompany(H) == X)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(And(accompany(L) == X, accompany(H) == X), (accompany(G) != X))
verification_results.append(result_0)
result_1 = is_deduced(And(accompany(L) == X, accompany(H) == X), (accompany(F) != X))
verification_results.append(result_1)
result_2 = is_deduced(And(accompany(L) == X, accompany(H) == X, accompany(G) != X), (accompany(M) == Y))
verification_results.append(result_2)
result_3 = is_deduced(accompany(H) == X, Or(accompany(H) == X, accompany(H) == Y))
verification_results.append(result_3)
result_4 = is_deduced(And(accompany(H) == X, accompany(M) == Y), (accompany(H) == X))
verification_results.append(result_4)
result_5 = is_deduced(And(accompany(L) == X, accompany(H) == X, accompany(M) == Y), (accompany(K) == W))
verification_results.append(result_5)

# Print all verification results
print('All verification results:', verification_results)
