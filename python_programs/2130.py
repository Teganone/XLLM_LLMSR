from z3 import *

candidates_sort, (ChengQiang, Julie, LiPing, XueFang) = EnumSort('candidates', ['ChengQiang', 'Julie', 'LiPing', 'XueFang'])
candidates = [ChengQiang, Julie, LiPing, XueFang]
meets_degree = Function('meets_degree', candidates_sort, BoolSort())
meets_english = Function('meets_english', candidates_sort, BoolSort())
meets_secretarial = Function('meets_secretarial', candidates_sort, BoolSort())
is_accepted = Function('is_accepted', candidates_sort, BoolSort())

pre_conditions = []
pre_conditions.append(Sum([meets_degree(c) == True for c in candidates]) == 3)
pre_conditions.append(Sum([meets_english(c) == True for c in candidates]) == 2)
pre_conditions.append(Sum([meets_secretarial(c) == True for c in candidates]) == 1)
c = Const('c', candidates_sort)
pre_conditions.append(ForAll([c], Or(meets_degree(c) == True, meets_english(c) == True, meets_secretarial(c) == True)))
pre_conditions.append(is_accepted(c) == And(meets_degree(c) == True, meets_english(c) == True, meets_secretarial(c) == True))
pre_conditions.append(Not(meets_degree(ChengQiang)))
pre_conditions.append(meets_degree(Julie))
pre_conditions.append(meets_degree(Julie) == meets_degree(XueFang))
pre_conditions.append(meets_english(LiPing) == True)
pre_conditions.append(meets_english(XueFang) == True)
pre_conditions.append(meets_secretarial(XueFang))
pre_conditions.append(Not(meets_secretarial(ChengQiang)))
pre_conditions.append(Not(meets_secretarial(Julie)))
pre_conditions.append(Not(meets_secretarial(LiPing)))
pre_conditions.append(Sum([is_accepted(c) == True for c in candidates]) == 1)
pre_conditions.append(Sum([meets_degree(c) == True for c in candidates]) == 3)
pre_conditions.append(Sum([meets_english(c) == True for c in candidates]) == 2)
pre_conditions.append(Sum([meets_secretarial(c) == True for c in candidates]) == 1)
c = Const('c', candidates_sort)
pre_conditions.append(ForAll([c], Or(meets_degree(c) == True, meets_english(c) == True, meets_secretarial(c) == True)))
pre_conditions.append(is_accepted(c) == And(meets_degree(c) == True, meets_english(c) == True, meets_secretarial(c) == True))
pre_conditions.append(Not(meets_degree(ChengQiang)))
pre_conditions.append(meets_degree(Julie))
pre_conditions.append(meets_degree(Julie) == meets_degree(XueFang))
pre_conditions.append(meets_english(LiPing) == True)
pre_conditions.append(meets_english(XueFang) == True)
pre_conditions.append(meets_secretarial(XueFang))
pre_conditions.append(Not(meets_secretarial(ChengQiang)))
pre_conditions.append(Not(meets_secretarial(Julie)))
pre_conditions.append(Not(meets_secretarial(LiPing)))
pre_conditions.append(Sum([is_accepted(c) == True for c in candidates]) == 1)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(And(Not(meets_degree(ChengQiang) == True), Not(meets_degree(Julie) == True)), True)
verification_results.append(result_0)
result_1 = is_deduced(meets_degree(Julie) == meets_degree(XueFang), True)
verification_results.append(result_1)
result_2 = is_deduced(And(meets_english(LiPing) == True, meets_english(XueFang) == True), True)
verification_results.append(result_2)
result_3 = is_deduced(And(meets_degree(Julie) == True, meets_english(Julie) == False, meets_degree(XueFang) == True, meets_english(XueFang) == True), True)
verification_results.append(result_3)
result_4 = is_deduced(meets_secretarial(XueFang) == True, True)
verification_results.append(result_4)

# Print all verification results
print('All verification results:', verification_results)
