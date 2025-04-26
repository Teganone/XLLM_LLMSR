from z3 import *

persons_sort, (XiaoWang, XiaoZhang, XiaoLi, XiaoZhao, XiaoZhou) = EnumSort('persons', ['XiaoWang', 'XiaoZhang', 'XiaoLi', 'XiaoZhao', 'XiaoZhou'])
persons = [XiaoWang, XiaoZhang, XiaoLi, XiaoZhao, XiaoZhou]
is_worker = Function('is_worker', persons_sort, BoolSort())
is_doctor = Function('is_doctor', persons_sort, BoolSort())
is_student = Function('is_student', persons_sort, BoolSort())
is_manager = Function('is_manager', persons_sort, BoolSort())

pre_conditions = []
pre_conditions.append(Implies(is_worker(XiaoWang) == True, is_doctor(XiaoZhang) == False))
pre_conditions.append(Or(is_worker(XiaoLi) == True, is_worker(XiaoWang) == True))
pre_conditions.append(Implies(is_doctor(XiaoZhang) == False, is_student(XiaoZhao) == False))
pre_conditions.append(Or(is_student(XiaoZhao) == True, is_manager(XiaoZhou) == False))
pre_conditions.append(Implies(is_worker(XiaoWang) == True, is_doctor(XiaoZhang) == False))
pre_conditions.append(Or(is_worker(XiaoLi) == True, is_worker(XiaoWang) == True))
pre_conditions.append(Implies(is_doctor(XiaoZhang) == False, is_student(XiaoZhao) == False))
pre_conditions.append(Or(is_student(XiaoZhao) == True, is_manager(XiaoZhou) == False))

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(And(Or(is_worker(XiaoLi) == True, is_worker(XiaoWang) == True), Not(is_worker(XiaoWang) == True)), is_worker(XiaoLi) == True)
verification_results.append(result_0)
result_1 = is_deduced(Not(is_worker(XiaoWang) == True), is_worker(XiaoLi) == True)
verification_results.append(result_1)

# Print all verification results
print('All verification results:', verification_results)
