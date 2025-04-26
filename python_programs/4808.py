from z3 import *

streets_sort, (Zhongshan, Yangtze, Meiyuan, Xinghai) = EnumSort('streets', ['Zhongshan', 'Yangtze', 'Meiyuan', 'Xinghai'])
streets = [Zhongshan, Yangtze, Meiyuan, Xinghai]
rank_of = Function('rank_of', streets_sort, IntSort())

pre_conditions = []
pre_conditions.append(Distinct([rank_of(s) for s in streets]))
pre_conditions.append(Implies(rank_of(Zhongshan) == 3, rank_of(Meiyuan) == 1))
pre_conditions.append(Implies(And(rank_of(Yangtze) != 1, rank_of(Yangtze) != 2), rank_of(Zhongshan) == 3))
pre_conditions.append(Or(And(rank_of(Zhongshan) == 3, rank_of(Meiyuan) == 3), And(rank_of(Zhongshan) != 3, rank_of(Meiyuan) != 3)))
pre_conditions.append(Distinct([rank_of(s) for s in streets]))
pre_conditions.append(Implies(rank_of(Zhongshan) == 3, rank_of(Meiyuan) == 1))
pre_conditions.append(Implies(And(rank_of(Yangtze) != 1, rank_of(Yangtze) != 2), rank_of(Zhongshan) == 3))
pre_conditions.append(Or(And(rank_of(Zhongshan) == 3, rank_of(Meiyuan) == 3), And(rank_of(Zhongshan) != 3, rank_of(Meiyuan) != 3)))

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(Implies(rank_of(Zhongshan) == 3, rank_of(Meiyuan) == 1), True)
verification_results.append(result_0)
result_1 = is_deduced(Implies(And(rank_of(Yangtze) != 1, rank_of(Yangtze) != 2), rank_of(Zhongshan) == 3), True)
verification_results.append(result_1)
result_2 = is_deduced(Or(And(rank_of(Zhongshan) == 3, rank_of(Meiyuan) == 3), And(rank_of(Zhongshan) != 3, rank_of(Meiyuan) != 3)), True)
verification_results.append(result_2)

# Print all verification results
print('All verification results:', verification_results)
