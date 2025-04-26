from z3 import *

persons_sort, (father, mother, son, daughter) = EnumSort('persons', ['father', 'mother', 'son', 'daughter'])
hobbies_sort, (music, sports, photography, reading) = EnumSort('hobbies', ['music', 'sports', 'photography', 'reading'])
subjects_sort, (mathematics, history, logic, physics) = EnumSort('subjects', ['mathematics', 'history', 'logic', 'physics'])
persons = [father, mother, son, daughter]
hobbies = [music, sports, photography, reading]
subjects = [mathematics, history, logic, physics]
seat = Function('seat', persons_sort, IntSort())
is_driver = Function('is_driver', persons_sort, BoolSort())
hobby = Function('hobby', persons_sort, hobbies_sort)
subject = Function('subject', persons_sort, subjects_sort)

pre_conditions = []
pre_conditions.append(Sum([is_driver(p) == True for p in persons]) == 1)
p = Const('p', persons_sort)
pre_conditions.append(ForAll([p], Implies(is_driver(p) == True, seat(p) <= 2)))
pre_conditions.append(Distinct([seat(p) for p in persons]))
pre_conditions.append(Distinct([hobby(p) for p in persons]))
pre_conditions.append(Distinct([subject(p) for p in persons]))
pre_conditions.append(subject(son) == physics)
pre_conditions.append(hobby(mother) == music)
p = Const('p', persons_sort)
pre_conditions.append(ForAll([p], Implies(And(hobby(p) == photography, subject(p) == logic), Or(And(seat(p) <= 2, seat(daughter) <= 2), And(seat(p) > 2, seat(daughter) > 2)))))
pre_conditions.append(hobby(son) != photography)
p = Const('p', persons_sort)
pre_conditions.append(ForAll([p], Implies(is_driver(p) == True, seat(son) == 5 - seat(p))))
pre_conditions.append(Or(And(seat(son) <= 2, seat(daughter) > 2), And(seat(son) > 2, seat(daughter) <= 2)))
p = Const('p', persons_sort)
pre_conditions.append(ForAll([p], Implies(And(p != daughter, hobby(p) == photography, subject(p) == logic, seat(p) <= 2), is_driver(p) == True)))
pre_conditions.append(hobby(father) == photography)
pre_conditions.append(subject(father) == logic)
pre_conditions.append(Sum([is_driver(p) == True for p in persons]) == 1)
p = Const('p', persons_sort)
pre_conditions.append(ForAll([p], Implies(is_driver(p) == True, seat(p) <= 2)))
pre_conditions.append(Distinct([seat(p) for p in persons]))
pre_conditions.append(Distinct([hobby(p) for p in persons]))
pre_conditions.append(Distinct([subject(p) for p in persons]))
pre_conditions.append(subject(son) == physics)
pre_conditions.append(hobby(mother) == music)
p = Const('p', persons_sort)
pre_conditions.append(ForAll([p], Implies(And(hobby(p) == photography, subject(p) == logic), Or(And(seat(p) <= 2, seat(daughter) <= 2), And(seat(p) > 2, seat(daughter) > 2)))))
pre_conditions.append(hobby(son) != photography)
p = Const('p', persons_sort)
pre_conditions.append(ForAll([p], Implies(is_driver(p) == True, seat(son) == 5 - seat(p))))
pre_conditions.append(Or(And(seat(son) <= 2, seat(daughter) > 2), And(seat(son) > 2, seat(daughter) <= 2)))
p = Const('p', persons_sort)
pre_conditions.append(ForAll([p], Implies(And(p != daughter, hobby(p) == photography, subject(p) == logic, seat(p) <= 2), is_driver(p) == True)))
pre_conditions.append(hobby(father) == photography)
pre_conditions.append(subject(father) == logic)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
p = Const('p', persons_sort)
result_0 = is_deduced(ForAll([p], Implies(is_driver(p) == True, And(seat(p) <= 2, seat(son) == 5 - seat(p)))), True)
verification_results.append(result_0)
result_1 = is_deduced(Or(And(seat(son) <= 2, seat(daughter) > 2), And(seat(son) > 2, seat(daughter) <= 2)), True)
verification_results.append(result_1)
p = Const('p', persons_sort)
result_2 = is_deduced(ForAll([p], Implies(And(hobby(p) == photography, subject(p) == logic), Or(And(seat(p) <= 2, seat(daughter) <= 2), And(seat(p) > 2, seat(daughter) > 2)))), True)
verification_results.append(result_2)
result_3 = is_deduced(hobby(son) != photography, True)
verification_results.append(result_3)
result_4 = is_deduced(is_driver(father) == True, True)
verification_results.append(result_4)

# Print all verification results
print('All verification results:', verification_results)
