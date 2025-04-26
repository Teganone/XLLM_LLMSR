from z3 import *

days_sort, (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday) = EnumSort('days', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
weather_sort, (sunny, rainy, cloudy) = EnumSort('weather', ['sunny', 'rainy', 'cloudy'])
days = [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
weather = [sunny, rainy, cloudy]
forecast = Function('forecast', days_sort, weather_sort)
umbrella = Function('umbrella', days_sort, BoolSort())

pre_conditions = []
pre_conditions.append(forecast(Thursday) == sunny)
pre_conditions.append(forecast(Monday) != rainy)
pre_conditions.append(forecast(Friday) == cloudy)
pre_conditions.append(Or(forecast(Friday) == rainy, forecast(Friday) == cloudy))
pre_conditions.append(Sum([forecast(d) == sunny for d in days]) == 3)
d = Const('d', days_sort)
pre_conditions.append(ForAll([d], umbrella(d) == (forecast(d) == rainy)))
pre_conditions.append(forecast(Thursday) == sunny)
pre_conditions.append(forecast(Monday) != rainy)
pre_conditions.append(forecast(Friday) == cloudy)
pre_conditions.append(Or(forecast(Friday) == rainy, forecast(Friday) == cloudy))
pre_conditions.append(Sum([forecast(d) == sunny for d in days]) == 3)
d = Const('d', days_sort)
pre_conditions.append(ForAll([d], umbrella(d) == (forecast(d) == rainy)))

def is_deduced(evidence, statement):
    solver = Solver()
    solver.add(evidence)
    solver.add(Not(statement))
    return solver.check() == unsat


verification_results = []

# Process verification blocks
result_0 = is_deduced(forecast(Friday) == sunny, And(forecast(Thursday) == sunny, forecast(Friday) == cloudy))
verification_results.append(result_0)
result_1 = is_deduced(forecast(Monday) == sunny, And(Sum([forecast(d) == sunny for d in days]) == 3, forecast(Thursday) == sunny, forecast(Friday) == sunny))
verification_results.append(result_1)
result_2 = is_deduced(umbrella(Tuesday) == True, True)
verification_results.append(result_2)

# Print all verification results
print('All verification results:', verification_results)
