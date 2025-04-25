from z3 import *

candidates_sort, (ChengQiang, Julie, LiPing, XueFang) = EnumSort('candidates', ['ChengQiang', 'Julie', 'LiPing', 'XueFang'])
candidates = [ChengQiang, Julie, LiPing, XueFang]
masters = Function('masters', candidates_sort, BoolSort())
english = Function('english', candidates_sort, BoolSort())
experience = Function('experience', candidates_sort, BoolSort())
accepted = Function('accepted', candidates_sort, BoolSort())

pre_conditions = []
pre_conditions.append(Sum([masters(c) == True for c in candidates]) == 3)
pre_conditions.append(Sum([english(c) == True for c in candidates]) == 2)
pre_conditions.append(Sum([experience(c) == True for c in candidates]) == 1)
c = Const('c', candidates_sort)
pre_conditions.append(ForAll([c], Or(masters(c) == True, english(c) == True, experience(c) == True)))
c = Const('c', candidates_sort)
pre_conditions.append(ForAll([c], accepted(c) == And(masters(c) == True, english(c) == True, experience(c) == True)))
pre_conditions.append(Sum([accepted(c) == True for c in candidates]) == 1)
pre_conditions.append(Not(masters(ChengQiang)))
pre_conditions.append(Not(masters(Julie)))
pre_conditions.append(masters(Julie) == True)
pre_conditions.append(masters(XueFang) == True)
pre_conditions.append(english(LiPing) == True)
pre_conditions.append(english(XueFang) == True)
pre_conditions.append(experience(XueFang) == True)
pre_conditions.append(Sum([masters(c) == True for c in candidates]) == 3)
pre_conditions.append(Sum([english(c) == True for c in candidates]) == 2)
pre_conditions.append(Sum([experience(c) == True for c in candidates]) == 1)
c = Const('c', candidates_sort)
pre_conditions.append(ForAll([c], Or(masters(c) == True, english(c) == True, experience(c) == True)))
c = Const('c', candidates_sort)
pre_conditions.append(ForAll([c], accepted(c) == And(masters(c) == True, english(c) == True, experience(c) == True)))
pre_conditions.append(Sum([accepted(c) == True for c in candidates]) == 1)
pre_conditions.append(Not(masters(ChengQiang)))
pre_conditions.append(Not(masters(Julie)))
pre_conditions.append(masters(Julie) == True)
pre_conditions.append(masters(XueFang) == True)
pre_conditions.append(english(LiPing) == True)
pre_conditions.append(english(XueFang) == True)
pre_conditions.append(experience(XueFang) == True)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(And(Not(masters(ChengQiang)), Not(masters(Julie))), True)
verification_results.append(result_0)
result_1 = is_deduced(And(masters(Julie) == True, masters(XueFang) == True), True)
verification_results.append(result_1)
result_2 = is_deduced(And(english(LiPing) == True, english(XueFang) == True), True)
verification_results.append(result_2)
result_3 = is_deduced(accepted(XueFang) == True, True)
verification_results.append(result_3)
result_4 = is_deduced(experience(XueFang) == True, True)
verification_results.append(result_4)

# Print all verification results
print('All verification results:', verification_results)
