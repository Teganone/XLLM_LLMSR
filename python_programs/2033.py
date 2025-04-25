from z3 import *

people_sort, (A, B, C, D, P) = EnumSort('people', ['A', 'B', 'C', 'D', 'P'])
seats_sort, (seatA, seatB, seatC, seatD, seatF) = EnumSort('seats', ['seatA', 'seatB', 'seatC', 'seatD', 'seatF'])
people = [A, B, C, D, P]
seats = [seatA, seatB, seatC, seatD, seatF]
seat_of = Function('seat_of', people_sort, seats_sort)

pre_conditions = []
pre_conditions.append(Distinct([seat_of(p) for p in people]))
pre_conditions.append(Implies(Or(seat_of(A) == seatC, seat_of(B) == seatC), seat_of(C) == seatB))
pre_conditions.append(Implies(seat_of(P) == seatC, seat_of(D) == seatF))
pre_conditions.append(seat_of(D) == seatB)
pre_conditions.append(Distinct([seat_of(p) for p in people]))
pre_conditions.append(Implies(Or(seat_of(A) == seatC, seat_of(B) == seatC), seat_of(C) == seatB))
pre_conditions.append(Implies(seat_of(P) == seatC, seat_of(D) == seatF))
pre_conditions.append(seat_of(D) == seatB)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(seat_of(D) == seatB, Not(seat_of(A) == seatC))
verification_results.append(result_0)
result_1 = is_deduced(seat_of(D) == seatB, seat_of(A) == seatA)
verification_results.append(result_1)

# Print all verification results
print('All verification results:', verification_results)
