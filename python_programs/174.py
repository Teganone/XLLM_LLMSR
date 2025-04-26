from z3 import *

recruits_sort, (F, G, H, I, W, X, Y) = EnumSort('recruits', ['F', 'G', 'H', 'I', 'W', 'X', 'Y'])
arms_sort, (communications, engineering, transport) = EnumSort('arms', ['communications', 'engineering', 'transport'])
recruits = [F, G, H, I, W, X, Y]
arms = [communications, engineering, transport]
arm_of = Function('arm_of', recruits_sort, arms_sort)

pre_conditions = []
pre_conditions.append(Sum([arm_of(r) == communications for r in recruits]) == 1)
pre_conditions.append(Sum([arm_of(r) == engineering for r in recruits]) == 3)
pre_conditions.append(Sum([arm_of(r) == transport for r in recruits]) == 3)
pre_conditions.append(arm_of(H) == arm_of(Y))
pre_conditions.append(arm_of(F) != arm_of(G))
pre_conditions.append(Implies(arm_of(X) == transport, arm_of(W) == engineering))
pre_conditions.append(arm_of(F) == engineering)
pre_conditions.append(arm_of(X) != engineering)
pre_conditions.append(Sum([arm_of(r) == communications for r in recruits]) == 1)
pre_conditions.append(Sum([arm_of(r) == engineering for r in recruits]) == 3)
pre_conditions.append(Sum([arm_of(r) == transport for r in recruits]) == 3)
pre_conditions.append(arm_of(H) == arm_of(Y))
pre_conditions.append(arm_of(F) != arm_of(G))
pre_conditions.append(Implies(arm_of(X) == transport, arm_of(W) == engineering))
pre_conditions.append(arm_of(F) == engineering)
pre_conditions.append(arm_of(X) != engineering)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(arm_of(X) != engineering, Or(arm_of(X) == communications, arm_of(X) == transport))
verification_results.append(result_0)
result_1 = is_deduced(arm_of(X) == transport, And(arm_of(W) == engineering, arm_of(H) == arm_of(Y)))
verification_results.append(result_1)
result_2 = is_deduced(True, arm_of(F) == engineering)
verification_results.append(result_2)
result_3 = is_deduced(arm_of(X) != engineering, And(arm_of(H) == transport, arm_of(W) == transport))
verification_results.append(result_3)

# Print all verification results
print('All verification results:', verification_results)
