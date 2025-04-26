from z3 import *


pre_conditions = []
pre_conditions.append(not(xiao_li == True))
pre_conditions.append(not(xiao_wang == True))
pre_conditions.append(implies(xiao_wang == True, xiao_miao == True))
pre_conditions.append(not(xiao_li == True))
pre_conditions.append(not(xiao_wang == True))
pre_conditions.append(implies(xiao_wang == True, xiao_miao == True))

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(True, not(xiao_li == True))
verification_results.append(result_0)
result_1 = is_deduced(True, not(xiao_wang == True))
verification_results.append(result_1)
result_2 = is_deduced(True, (implies(xiao_wang == True, xiao_miao == True) or (xiao_li == True)))
verification_results.append(result_2)

# Print all verification results
print('All verification results:', verification_results)
