from z3 import *

candidates_sort, (ChengQiang, Julie, LiPing, XueFang) = EnumSort('candidates', ['ChengQiang', 'Julie', 'LiPing', 'XueFang'])
candidates = [ChengQiang, Julie, LiPing, XueFang]
meets_master = Function('meets_master', candidates_sort, BoolSort())
meets_english = Function('meets_english', candidates_sort, BoolSort())
meets_secretarial = Function('meets_secretarial', candidates_sort, BoolSort())

pre_conditions = []
pre_conditions.append(Sum([meets_master(p)==True for p in candidates]) == 3)
pre_conditions.append(Sum([meets_english(p)==True for p in candidates]) == 2)
pre_conditions.append(Sum([meets_secretarial(p)==True for p in candidates]) == 1)
p = Const('p', candidates_sort)
pre_conditions.append(ForAll([p], Or(meets_master(p)==True, meets_english(p)==True, meets_secretarial(p)==True)))
pre_conditions.append(Not(And(meets_master(ChengQiang)==True, meets_master(Julie)==True)))
pre_conditions.append((meets_master(Julie) == meets_master(XueFang)))
pre_conditions.append((meets_master(Julie)==True))
pre_conditions.append((meets_english(LiPing) == meets_english(XueFang)))
pre_conditions.append((meets_english(Julie)==True))
pre_conditions.append(Sum([And(meets_master(p)==True, meets_english(p)==True, meets_secretarial(p)==True) for p in candidates]) == 1)
pre_conditions.append(Sum([meets_master(p)==True for p in candidates]) == 3)
pre_conditions.append(Sum([meets_english(p)==True for p in candidates]) == 2)
pre_conditions.append(Sum([meets_secretarial(p)==True for p in candidates]) == 1)
p = Const('p', candidates_sort)
pre_conditions.append(ForAll([p], Or(meets_master(p)==True, meets_english(p)==True, meets_secretarial(p)==True)))
pre_conditions.append(Not(And(meets_master(ChengQiang)==True, meets_master(Julie)==True)))
pre_conditions.append((meets_master(Julie) == meets_master(XueFang)))
pre_conditions.append((meets_master(Julie)==True))
pre_conditions.append((meets_english(LiPing) == meets_english(XueFang)))
pre_conditions.append((meets_english(Julie)==True))
pre_conditions.append(Sum([And(meets_master(p)==True, meets_english(p)==True, meets_secretarial(p)==True) for p in candidates]) == 1)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(Not(And(meets_master(ChengQiang)==True, meets_master(Julie)==True)), True)
verification_results.append(result_0)
result_1 = is_deduced(And(meets_master(Julie)==True, meets_master(XueFang)==True, meets_english(Julie)==True, meets_english(XueFang)==True), True)
verification_results.append(result_1)
result_2 = is_deduced(Sum([And(meets_master(p)==True, meets_english(p)==True, meets_secretarial(p)==True) for p in candidates]) == 1, And(meets_master(XueFang)==True, meets_english(XueFang)==True, meets_secretarial(XueFang)==True))
verification_results.append(result_2)

# Print all verification results
print('All verification results:', verification_results)
