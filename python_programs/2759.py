from z3 import *

persons_sort, (A, B, C) = EnumSort('persons', ['A', 'B', 'C'])
occupations_sort, (lawyer, doctor, teacher) = EnumSort('occupations', ['lawyer', 'doctor', 'teacher'])
persons = [A, B, C]
occupations = [lawyer, doctor, teacher]
occupation = Function('occupation', persons_sort, occupations_sort)
income = Function('income', persons_sort, IntSort())

pre_conditions = []
pre_conditions.append(Distinct([occupation(p) for p in persons]))
x = Const('x', persons_sort)
pre_conditions.append(ForAll([x], Implies(And(occupation(x) == teacher, x != C), income(C) > income(x))))
x = Const('x', persons_sort)
pre_conditions.append(ForAll([x], Implies(And(occupation(x) == doctor, x != A), income(A) != income(x))))
x = Const('x', persons_sort)
pre_conditions.append(ForAll([x], Implies(occupation(x) == doctor, income(x) < income(B))))
pre_conditions.append(Distinct([occupation(p) for p in persons]))
x = Const('x', persons_sort)
pre_conditions.append(ForAll([x], Implies(And(occupation(x) == teacher, x != C), income(C) > income(x))))
x = Const('x', persons_sort)
pre_conditions.append(ForAll([x], Implies(And(occupation(x) == doctor, x != A), income(A) != income(x))))
x = Const('x', persons_sort)
pre_conditions.append(ForAll([x], Implies(occupation(x) == doctor, income(x) < income(B))))

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
x = Const('x', persons_sort)
result_0 = is_deduced(ForAll([x], Implies(And(occupation(x) == teacher, x != C), income(C) > income(x))), True)
verification_results.append(result_0)
x = Const('x', persons_sort)
result_1 = is_deduced(ForAll([x], Implies(And(occupation(x) == doctor, x != A), income(A) != income(x))), True)
verification_results.append(result_1)
x = Const('x', persons_sort)
result_2 = is_deduced(ForAll([x], Implies(occupation(x) == doctor, income(x) < income(B))), True)
verification_results.append(result_2)
result_3 = is_deduced(And(occupation(A) == doctor, occupation(B) != doctor, occupation(C) == teacher), True)
verification_results.append(result_3)

# Print all verification results
print('All verification results:', verification_results)
