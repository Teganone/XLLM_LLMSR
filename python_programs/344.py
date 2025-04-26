from z3 import *

employees_sort, (A, B, C) = EnumSort('employees', ['A', 'B', 'C'])
awards_sort, (Professional, Creative, Collaboration, Writing, Star) = EnumSort('awards', ['Professional', 'Creative', 'Collaboration', 'Writing', 'Star'])
employees = [A, B, C]
awards = [Professional, Creative, Collaboration, Writing, Star]
award_winner = Function('award_winner', awards_sort, employees_sort)

pre_conditions = []
e = Const('e', employees_sort)
pre_conditions.append(ForAll([e], Sum([award_winner(aw) == e for aw in awards]) <= 2))
pre_conditions.append(award_winner(Star) == A)
pre_conditions.append(Not(Or(award_winner(Creative) == A, award_winner(Professional) == A)))
pre_conditions.append(Not(award_winner(Collaboration) == award_winner(Writing)))
pre_conditions.append(Not(award_winner(Professional) == award_winner(Collaboration)))
pre_conditions.append(award_winner(Creative) == B)
pre_conditions.append(award_winner(Collaboration) == C)
pre_conditions.append(Not(award_winner(Professional) == B))
pre_conditions.append(Or(award_winner(Writing) == B, award_winner(Writing) == C))
pre_conditions.append(Sum([award_winner(aw) == B for aw in awards]) == 1)
pre_conditions.append(Sum([award_winner(aw) == C for aw in awards]) == 1)
e = Const('e', employees_sort)
pre_conditions.append(ForAll([e], Sum([award_winner(aw) == e for aw in awards]) <= 2))
pre_conditions.append(award_winner(Star) == A)
pre_conditions.append(Not(Or(award_winner(Creative) == A, award_winner(Professional) == A)))
pre_conditions.append(Not(award_winner(Collaboration) == award_winner(Writing)))
pre_conditions.append(Not(award_winner(Professional) == award_winner(Collaboration)))
pre_conditions.append(award_winner(Creative) == B)
pre_conditions.append(award_winner(Collaboration) == C)
pre_conditions.append(Not(award_winner(Professional) == B))
pre_conditions.append(Or(award_winner(Writing) == B, award_winner(Writing) == C))
pre_conditions.append(Sum([award_winner(aw) == B for aw in awards]) == 1)
pre_conditions.append(Sum([award_winner(aw) == C for aw in awards]) == 1)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(award_winner(Star) == A, Not(Or(award_winner(Creative) == A, award_winner(Professional) == A)))
verification_results.append(result_0)
result_1 = is_deduced(True, Sum([award_winner(aw) == B for aw in awards]) == 1)
verification_results.append(result_1)
result_2 = is_deduced(True, Sum([award_winner(aw) == C for aw in awards]) == 1)
verification_results.append(result_2)
result_3 = is_deduced(True, award_winner(Collaboration) != B)
verification_results.append(result_3)
result_4 = is_deduced(True, award_winner(Professional) != B)
verification_results.append(result_4)
result_5 = is_deduced(True, award_winner(Creative) == B)
verification_results.append(result_5)
result_6 = is_deduced(True, award_winner(Collaboration) == C)
verification_results.append(result_6)
result_7 = is_deduced(award_winner(Collaboration) == C, award_winner(Professional) != B)
verification_results.append(result_7)

# Print all verification results
print('All verification results:', verification_results)
