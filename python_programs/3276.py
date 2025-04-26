from z3 import *

teachers_sort, (Cai, Zhu, Sun) = EnumSort('teachers', ['Cai', 'Zhu', 'Sun'])
subjects_sort, (biology, physics, English, politics, history, mathematics) = EnumSort('subjects', ['biology', 'physics', 'English', 'politics', 'history', 'mathematics'])
teachers = [Cai, Zhu, Sun]
subjects = [biology, physics, English, politics, history, mathematics]
subject1 = Function('subject1', teachers_sort, subjects_sort)
subject2 = Function('subject2', teachers_sort, subjects_sort)
age_of = Function('age_of', teachers_sort, IntSort())

pre_conditions = []
pre_conditions.append(Distinct([age_of(t) for t in teachers]))
t = Const('t', teachers_sort)
pre_conditions.append(ForAll([t], subject1(t) != subject2(t)))
pre_conditions.append(Or(subject1(Cai) == mathematics, subject2(Cai) == mathematics))
pre_conditions.append(Or(subject1(Sun) == physics, subject2(Sun) == physics))
pre_conditions.append(Or(subject1(Zhu) == politics, subject2(Zhu) == politics))
pre_conditions.append(Not(Or(subject1(Cai) == English, subject2(Cai) == English)))
pre_conditions.append(Not(Or(subject1(Sun) == politics, subject2(Sun) == politics)))
pre_conditions.append(And(age_of(Cai) < age_of(Zhu), age_of(Cai) < age_of(Sun)))
pre_conditions.append(Or(age_of(Sun) == age_of(Zhu) + 1, age_of(Sun) == age_of(Zhu) - 1))
t1 = Const('t1', teachers_sort)
t2 = Const('t2', teachers_sort)
pre_conditions.append(ForAll([t1, t2], Implies( And( Or(subject1(t1) == biology, subject2(t1) == biology), Or(subject1(t2) == mathematics, subject2(t2) == mathematics) ), age_of(t2) > age_of(t1) )))
pre_conditions.append(Distinct([age_of(t) for t in teachers]))
t = Const('t', teachers_sort)
pre_conditions.append(ForAll([t], subject1(t) != subject2(t)))
pre_conditions.append(Or(subject1(Cai) == mathematics, subject2(Cai) == mathematics))
pre_conditions.append(Or(subject1(Sun) == physics, subject2(Sun) == physics))
pre_conditions.append(Or(subject1(Zhu) == politics, subject2(Zhu) == politics))
pre_conditions.append(Not(Or(subject1(Cai) == English, subject2(Cai) == English)))
pre_conditions.append(Not(Or(subject1(Sun) == politics, subject2(Sun) == politics)))
pre_conditions.append(And(age_of(Cai) < age_of(Zhu), age_of(Cai) < age_of(Sun)))
pre_conditions.append(Or(age_of(Sun) == age_of(Zhu) + 1, age_of(Sun) == age_of(Zhu) - 1))
t1 = Const('t1', teachers_sort)
t2 = Const('t2', teachers_sort)
pre_conditions.append(ForAll([t1, t2], Implies( And( Or(subject1(t1) == biology, subject2(t1) == biology), Or(subject1(t2) == mathematics, subject2(t2) == mathematics) ), age_of(t2) > age_of(t1) )))

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(And(age_of(Cai) < age_of(Zhu), age_of(Cai) < age_of(Sun)), True)
verification_results.append(result_0)
result_1 = is_deduced(Or(subject1(Cai) == mathematics, subject2(Cai) == mathematics), True)
verification_results.append(result_1)
result_2 = is_deduced(And(Or(subject1(Cai) == mathematics, subject2(Cai) == mathematics), Not(Or(subject1(Cai) == English, subject2(Cai) == English))), True)
verification_results.append(result_2)
result_3 = is_deduced(Or(subject1(Sun) == physics, subject2(Sun) == physics), True)
verification_results.append(result_3)
result_4 = is_deduced(Not(Or(subject1(Sun) == politics, subject2(Sun) == politics)), True)
verification_results.append(result_4)
result_5 = is_deduced(Or(subject1(Zhu) == politics, subject2(Zhu) == politics), True)
verification_results.append(result_5)

# Print all verification results
print('All verification results:', verification_results)
