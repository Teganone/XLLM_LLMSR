from z3 import *

materials_sort, (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30) = EnumSort('materials', ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'T25', 'T26', 'T27', 'T28', 'T29', 'T30'])
materials = [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30]
financial = Function('financial', materials_sort, BoolSort())
english = Function('english', materials_sort, BoolSort())
us_import = Function('us_import', materials_sort, BoolSort())

pre_conditions = []
pre_conditions.append(Sum([True for m in materials]) == 30)
pre_conditions.append(Sum([financial(m) == True for m in materials]) == 12)
pre_conditions.append(Sum([And(financial(m) == False, english(m) == True) for m in materials]) == 10)
pre_conditions.append(Sum([And(financial(m) == False, us_import(m) == True) for m in materials]) == 7)
pre_conditions.append(Sum([And(english(m) == False, us_import(m) == False) for m in materials]) == 9)
pre_conditions.append(Sum([True for m in materials]) == 30)
pre_conditions.append(Sum([financial(m) == True for m in materials]) == 12)
pre_conditions.append(Sum([And(financial(m) == False, english(m) == True) for m in materials]) == 10)
pre_conditions.append(Sum([And(financial(m) == False, us_import(m) == True) for m in materials]) == 7)
pre_conditions.append(Sum([And(english(m) == False, us_import(m) == False) for m in materials]) == 9)

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(Sum([True for m in materials]) == 30, True)
verification_results.append(result_0)
result_1 = is_deduced(Sum([financial(m) == True for m in materials]) == 12, True)
verification_results.append(result_1)
result_2 = is_deduced(Sum([And(financial(m) == False, english(m) == True) for m in materials]) == 10, True)
verification_results.append(result_2)
result_3 = is_deduced(Sum([And(financial(m) == False, us_import(m) == True) for m in materials]) == 7, True)
verification_results.append(result_3)
result_4 = is_deduced(Sum([And(english(m) == False, us_import(m) == False) for m in materials]) == 9, True)
verification_results.append(result_4)

# Print all verification results
print('All verification results:', verification_results)
