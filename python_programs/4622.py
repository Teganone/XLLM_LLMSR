from z3 import *

students_sort, (A, B, C, D) = EnumSort('students', ['A', 'B', 'C', 'D'])
universities_sort, (Peking, Tsinghua, Nanjing, Southeast) = EnumSort('universities', ['Peking', 'Tsinghua', 'Nanjing', 'Southeast'])
students = [A, B, C, D]
universities = [Peking, Tsinghua, Nanjing, Southeast]
attends = Function('attends', students_sort, universities_sort)

pre_conditions = []
pre_conditions.append(Distinct([attends(s) for s in students]))
pre_conditions.append(attends(A) != Peking)
pre_conditions.append(attends(B) != Tsinghua)
pre_conditions.append(attends(C) != Nanjing)
pre_conditions.append(attends(D) != Southeast)
pre_conditions.append(Distinct([attends(s) for s in students]))
pre_conditions.append(attends(A) != Peking)
pre_conditions.append(attends(B) != Tsinghua)
pre_conditions.append(attends(C) != Nanjing)
pre_conditions.append(attends(D) != Southeast)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(attends(A) == Southeast, And(attends(A) != Peking, attends(C) != Nanjing))
verification_results.append(result_0)
result_1 = is_deduced(attends(C) == Tsinghua, And(attends(A) != Peking, attends(C) != Nanjing))
verification_results.append(result_1)
result_2 = is_deduced(attends(B) == Nanjing, And(attends(B) != Tsinghua, attends(A) != Peking))
verification_results.append(result_2)
result_3 = is_deduced(attends(D) == Peking, And(attends(D) != Southeast, attends(B) != Tsinghua))
verification_results.append(result_3)

# Print all verification results
print('All verification results:', verification_results)
