from z3 import *

candidates_sort, (Cheng_Qiang, Julie, Li_Ping, Xue_Fang) = EnumSort('candidates', ['Cheng_Qiang', 'Julie', 'Li_Ping', 'Xue_Fang'])
candidates = [Cheng_Qiang, Julie, Li_Ping, Xue_Fang]
qualifies1 = Function('qualifies1', candidates_sort, BoolSort())
qualifies2 = Function('qualifies2', candidates_sort, BoolSort())
qualifies3 = Function('qualifies3', candidates_sort, BoolSort())

pre_conditions = []
pre_conditions.append(Sum([qualifies1(c) == True for c in candidates]) == 3)
pre_conditions.append(Sum([qualifies2(c) == True for c in candidates]) == 2)
pre_conditions.append(Sum([qualifies3(c) == True for c in candidates]) == 1)
c = Const('c', candidates_sort)
pre_conditions.append(ForAll([c], Or(qualifies1(c), qualifies2(c), qualifies3(c)) == True))
pre_conditions.append(((Not(qualifies1(Cheng_Qiang)) and qualifies1(Julie)) or (qualifies1(Cheng_Qiang) and Not(qualifies1(Julie)))))
pre_conditions.append(qualifies1(Julie) == qualifies1(Xue_Fang))
pre_conditions.append(qualifies2(Li_Ping) == qualifies2(Xue_Fang))
pre_conditions.append(qualifies1(Julie) == True)
pre_conditions.append(qualifies1(Xue_Fang) == True)
pre_conditions.append(qualifies2(Li_Ping) == True)
pre_conditions.append(qualifies2(Xue_Fang) == True)
pre_conditions.append(Sum([qualifies1(c) == True for c in candidates]) == 3)
pre_conditions.append(Sum([qualifies2(c) == True for c in candidates]) == 2)
pre_conditions.append(Sum([qualifies3(c) == True for c in candidates]) == 1)
c = Const('c', candidates_sort)
pre_conditions.append(ForAll([c], Or(qualifies1(c), qualifies2(c), qualifies3(c)) == True))
pre_conditions.append(((Not(qualifies1(Cheng_Qiang)) and qualifies1(Julie)) or (qualifies1(Cheng_Qiang) and Not(qualifies1(Julie)))))
pre_conditions.append(qualifies1(Julie) == qualifies1(Xue_Fang))
pre_conditions.append(qualifies2(Li_Ping) == qualifies2(Xue_Fang))
pre_conditions.append(qualifies1(Julie) == True)
pre_conditions.append(qualifies1(Xue_Fang) == True)
pre_conditions.append(qualifies2(Li_Ping) == True)
pre_conditions.append(qualifies2(Xue_Fang) == True)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(And(Not(qualifies1(Cheng_Qiang)), Not(qualifies1(Julie))), False)
verification_results.append(result_0)
result_1 = is_deduced(And(qualifies1(Julie), qualifies1(Xue_Fang)), True)
verification_results.append(result_1)
result_2 = is_deduced(And(qualifies2(Li_Ping), qualifies2(Xue_Fang)), True)
verification_results.append(result_2)
c = Const('c', candidates_sort)
result_3 = is_deduced(ForAll([c], Implies(And(qualifies1(c), qualifies2(c), qualifies3(c)), c == Xue_Fang)), True)
verification_results.append(result_3)

# Print all verification results
print('All verification results:', verification_results)
