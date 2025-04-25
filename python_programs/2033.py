from z3 import *

persons_sort, (A, B, C, D, P) = EnumSort('persons', ['A', 'B', 'C', 'D', 'P'])
seats_sort, (seatA, seatB, seatC, seatD, seatF) = EnumSort('seats', ['seatA', 'seatB', 'seatC', 'seatD', 'seatF'])
persons = [A, B, C, D, P]
seats = [seatA, seatB, seatC, seatD, seatF]
sits_in = Function('sits_in', persons_sort, seats_sort)

pre_conditions = []
pre_conditions.append(Distinct([sits_in(p) for p in persons]))
pre_conditions.append(Implies(Or(sits_in(A) == seatC, sits_in(B) == seatC), sits_in(C) == seatB))
pre_conditions.append(Implies(sits_in(P) == seatC, sits_in(D) == seatF))
pre_conditions.append(sits_in(D) == seatB)
pre_conditions.append(Distinct([sits_in(p) for p in persons]))
pre_conditions.append(Implies(Or(sits_in(A) == seatC, sits_in(B) == seatC), sits_in(C) == seatB))
pre_conditions.append(Implies(sits_in(P) == seatC, sits_in(D) == seatF))
pre_conditions.append(sits_in(D) == seatB)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(sits_in(D) == seatB, Not(sits_in(A) == seatC))
verification_results.append(result_0)
result_1 = is_deduced(sits_in(D) == seatB, sits_in(A) == seatA)
verification_results.append(result_1)

# Print all verification results
print('All verification results:', verification_results)
