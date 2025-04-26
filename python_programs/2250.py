from z3 import *

players_sort, (A, B, C) = EnumSort('players', ['A', 'B', 'C'])
teams_sort, (football, table_tennis, basketball) = EnumSort('teams', ['football', 'table_tennis', 'basketball'])
players = [A, B, C]
teams = [football, table_tennis, basketball]
team_of = Function('team_of', players_sort, teams_sort)

pre_conditions = []
pre_conditions.append(Distinct([team_of(p) for p in players]))
pre_conditions.append(team_of(A) == football)
pre_conditions.append(team_of(B) != football)
pre_conditions.append(team_of(C) != basketball)
pre_conditions.append(Distinct([team_of(p) for p in players]))
pre_conditions.append(team_of(A) == football)
pre_conditions.append(team_of(B) != football)
pre_conditions.append(team_of(C) != basketball)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(team_of(A) == football, team_of(A) == football)
verification_results.append(result_0)
result_1 = is_deduced(team_of(B) != football, team_of(B) != football)
verification_results.append(result_1)
result_2 = is_deduced(team_of(C) != basketball, team_of(C) != basketball)
verification_results.append(result_2)
result_3 = is_deduced(team_of(C) == table_tennis, team_of(C) == table_tennis)
verification_results.append(result_3)
result_4 = is_deduced(team_of(B) == basketball, team_of(B) == basketball)
verification_results.append(result_4)

# Print all verification results
print('All verification results:', verification_results)
