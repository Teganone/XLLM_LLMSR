from src.z3_solver.smt_solver import LLMSR_Z3_Program


def extract_verification_results(output):
    """
    从输出中提取验证结果列表
    
    Args:
        output: 执行逻辑程序的输出
        
    Returns:
        验证结果列表（布尔值）
    """
    # 查找包含"All verification results:"的行
    for line in output:
        if "All verification results:" in line:
            import re
            match = re.search(r'\[(.*?)\]', line)
            if match:
                # 提取列表内容并分割
                results_str = match.group(1)
                verification_results = [r.strip() for r in results_str.split(', ')]
                # print(verification_results)
                return verification_results
            # 提取[]中的内容
            start_idx = line.find('[')
            end_idx = line.find(']')
            if start_idx != -1 and end_idx != -1:
                # 提取列表内容
                results_str = line[start_idx+1:end_idx]
                # 将字符串转换为Python列表
                results = []
                for item in results_str.split():
                    if item.lower() == "true":
                        results.append(True)
                    elif item.lower() == "false":
                        results.append(False)
                return results
    
    return []  # 如果没有找到结果，返回空列表


logic_program = '''
# Declarations
students = EnumSort([G, H, L, M, U, W, Z])
countries = EnumSort([UK, US])
goes_to = Function([students] -> [countries])

# Constraints
Implies(goes_to(G) == UK, goes_to(H) == US) ::: (1) If G goes to the UK, then H To the United States
Implies(goes_to(L) == UK, And(goes_to(M) == US, goes_to(U) == US)) ::: (2) If L goes to the UK, both M and U go to the US
goes_to(W) != goes_to(Z) ::: (3) The country W went to was different from the country Z went to
goes_to(U) != goes_to(G) ::: (4) The country where U goes is different from the country where G goes
Implies(goes_to(Z) == UK, goes_to(H) == UK) ::: (5) If Z goes to the UK, then H also goes to the UK
goes_to(G) == US ::: If G goes to the United States

# Verifications
is_deduced(goes_to(G) == US, goes_to(G) != UK) ::: (1) Condition (1) is not applicable
is_deduced(goes_to(G) == US, True) ::: (2) Condition (2) is also not applicable
is_deduced(goes_to(W) != goes_to(Z), False) ::: (3) Condition (3) does not provide any information about H, M, U, or W
is_deduced(And(goes_to(U) != goes_to(G), goes_to(G) == US), goes_to(U) == UK) ::: (4) U must go to the UK
is_deduced(goes_to(G) == US, True) ::: (5) Condition (5) is not applicable
'''

# 更复杂的逻辑程序示例
logic_program_1 = '''
# Declarations
magicians = EnumSort([G, H, K, L, N, P, Q])
teams = EnumSort([T1, T2, TN])
positions = EnumSort([PF, PM, PB, PN])
team_of = Function([magicians] -> [teams])
position_of = Function([magicians] -> [positions])
is_playing = Function([magicians] -> [bool])

# Constraints
Count([m:magicians], is_playing(m)) == 6 ::: Choose 6 people to play
And(Count([m:magicians], team_of(m) == T1) == 3, Count([m:magicians], team_of(m) == T2) == 3) ::: Each team has exactly 3 magicians
And(Count([m:magicians], And(team_of(m) == T1, position_of(m) == PF)) == 1, Count([m:magicians], And(team_of(m) == T1, position_of(m) == PM)) == 1, Count([m:magicians], And(team_of(m) == T1, position_of(m) == PB)) == 1, Count([m:magicians], And(team_of(m) == T2, position_of(m) == PF)) == 1, Count([m:magicians], And(team_of(m) == T2, position_of(m) == PM)) == 1, Count([m:magicians], And(team_of(m) == T2, position_of(m) == PB)) == 1) ::: Each position in each team is occupied by exactly one magician
ForAll([m:magicians], is_playing(m) == And(team_of(m) != TN, position_of(m) != PN)) ::: A magician is playing if and only if they are assigned to a team and a position
ForAll([m:magicians], Not(is_playing(m)) == And(team_of(m) == TN, position_of(m) == PN)) ::: If a magician is not playing, they are assigned to no team and no position
And(Implies(is_playing(G), position_of(G) == PF), Implies(is_playing(H), position_of(H) == PF)) ::: (1) If G or H is arranged to play, they must be in the front
Implies(is_playing(K), position_of(K) == PM) ::: (2) If K is scheduled to play, he must be in the middle
Implies(is_playing(L), team_of(L) == T1) ::: (3) If L is scheduled to play, he must be on team 1
And(Implies(And(is_playing(P), is_playing(N)), team_of(P) != team_of(N)), Implies(And(is_playing(K), is_playing(N)), team_of(K) != team_of(N))) ::: (4) Neither P nor K can be in the same team as N
Implies(And(is_playing(P), is_playing(Q)), team_of(P) != team_of(Q)) ::: (5) P cannot be in the same team as Q
Implies(And(is_playing(H), team_of(H) == T2), And(is_playing(Q), team_of(Q) == T1, position_of(Q) == PM)) ::: (6) If H is in team 2, Q is in the middle of team 1

# Verifications
is_deduced(And(Implies(is_playing(G), position_of(G) == PF), Implies(is_playing(H), position_of(H) == PF), Implies(is_playing(L), team_of(L) == T1), is_playing(L), position_of(L) != PF), Or(position_of(L) == PM, position_of(L) == PB)) ::: (1) he must be in the middle or back
is_deduced(And(Implies(is_playing(K), position_of(K) == PM), is_playing(L), position_of(L) == PM), position_of(K) != PM) ::: (2) K must not be in the middle
is_deduced(And(Implies(And(is_playing(P), is_playing(N)), team_of(P) != team_of(N)), Implies(And(is_playing(K), is_playing(N)), team_of(K) != team_of(N)), is_playing(K), position_of(K) == PB), Not(position_of(P) == PB)) ::: (3) P cannot be in the back
is_deduced(And(Implies(And(is_playing(P), is_playing(N)), team_of(P) != team_of(N)), Implies(And(is_playing(K), is_playing(N)), team_of(K) != team_of(N)), is_playing(K), position_of(K) == PB, is_playing(P)), Or(position_of(P) == PF, position_of(P) == PM)) ::: (4) P must be in the front or middle
is_deduced(Implies(And(is_playing(P), is_playing(Q)), team_of(P) != team_of(Q)), Implies(And(is_playing(P), is_playing(L)), team_of(P) != team_of(L))) ::: (5) P cannot be in the same team as L
is_deduced(And(Implies(And(is_playing(P), is_playing(N)), team_of(P) != team_of(N)), Implies(And(is_playing(K), is_playing(N)), team_of(K) != team_of(N)), is_playing(K), position_of(K) == PB, is_playing(P), Implies(And(is_playing(P), is_playing(Q)), team_of(P) != team_of(Q)), team_of(L) == team_of(Q)), position_of(P) == PF) ::: (6) P must be in the front
is_deduced(And(Implies(And(is_playing(H), team_of(H) == T2), And(is_playing(Q), team_of(Q) == T1, position_of(Q) == PM)), is_playing(G), position_of(G) == PF, team_of(G) == T1), team_of(H) == T2) ::: (7) H must be in team 2
'''


logic_program_2 = '''
# Declarations
magicians = EnumSort([G, H, K, L, N, P, Q]) 
positions = EnumSort([front, middle, back, no_position]) 
teams = EnumSort([team1, team2, no_team])
team_of = Function([magicians] -> [teams]) 
position_of = Function([magicians] -> [positions]) 
is_playing = Function([magicians] -> [bool]) 

# Constraints
Count([m:magicians], is_playing(m)) == 6 ::: Choose 6 people to play
And(Count([m:magicians], team_of(m) == team1) == 3, Count([m:magicians], team_of(m) == team2) == 3) ::: # Each team has exactly 3 magicians
And(Count([m:magicians], And(team_of(m) == team1, position_of(m) == front)) == 1, Count([m:magicians], And(team_of(m) == team1, position_of(m) == middle)) == 1, Count([m:magicians], And(team_of(m) == team1, position_of(m) == back)) == 1, Count([m:magicians], And(team_of(m) == team2, position_of(m) == front)) == 1, Count([m:magicians], And(team_of(m) == team2, position_of(m) == middle)) == 1, Count([m:magicians], And(team_of(m) == team2, position_of(m) == back)) == 1) ::: Each position in each team is occupied by exactly one magician
ForAll([m:magicians], is_playing(m) == And(team_of(m) != no_team, position_of(m) != no_position)) ::: A magician is playing if and only if they are assigned to a team and a position
ForAll([m:magicians], Not(is_playing(m)) == And(team_of(m) == no_team, position_of(m) == no_position)) ::: If a magician is not playing, they are assigned to no team and no position
And(Implies(is_playing(G), position_of(G) == front), Implies(is_playing(H), position_of(H) == front)) ::: (1) If G or H is arranged to play, they must be in the front
Implies(is_playing(K), position_of(K) == middle) ::: (2) If K is scheduled to play, he must be in the middle
Implies(is_playing(L), team_of(L) == team1) ::: (3) If L is scheduled to play, he must be on team 1
And(Implies(And(is_playing(P), is_playing(N)), team_of(P) != team_of(N)), Implies(And(is_playing(K), is_playing(N)), team_of(K) != team_of(N))) ::: (4) Neither P nor K can be in the same team as N
Implies(And(is_playing(P), is_playing(Q)), team_of(P) != team_of(Q)) ::: (5) P cannot be in the same team as Q
Implies(And(is_playing(H), team_of(H) == team2), And(is_playing(Q), team_of(Q) == team1, position_of(Q) == middle)) ::: (6) If H is in team 2, Q is in the middle of team 1

# Verifications
is_deduced(And(Implies(is_playing(G), position_of(G) == front), Implies(is_playing(H), position_of(H) == front), Implies(is_playing(L), team_of(L) == team1), is_playing(L), position_of(L) != front), Or(position_of(L) == middle, position_of(L) == back)) ::: (1) he must be in the middle or back
is_deduced(And(Implies(is_playing(K), position_of(K) == middle), is_playing(L), position_of(L) == middle), position_of(K) == back) ::: (2) K must be in the back ::: (2) K must be in the back
is_deduced(And(Implies(And(is_playing(P), is_playing(N)), team_of(P) != team_of(N)), Implies(And(is_playing(K), is_playing(N)), team_of(K) != team_of(N)), is_playing(K), position_of(K) == back), Not(position_of(P) == back)) ::: (3) P cannot be in the back
is_deduced(And(Implies(And(is_playing(P), is_playing(N)), team_of(P) != team_of(N)), Implies(And(is_playing(K), is_playing(N)), team_of(K) != team_of(N)), is_playing(K), position_of(K) == back, is_playing(P)), Or(position_of(P) == front, position_of(P) == middle)) ::: (4) P must be in the front or middle
is_deduced(Implies(And(is_playing(P), is_playing(Q)), team_of(P) != team_of(Q)), Implies(And(is_playing(P), is_playing(L)), team_of(P) != team_of(L))) ::: (5) P cannot be in the same team as L
is_deduced(And(Implies(And(is_playing(P), is_playing(N)), team_of(P) != team_of(N)), Implies(And(is_playing(K), is_playing(N)), team_of(K) != team_of(N)), is_playing(K), position_of(K) == back, is_playing(P), Implies(And(is_playing(P), is_playing(Q)), team_of(P) != team_of(Q)), team_of(L) == team_of(Q)), position_of(P) == front) ::: (6) P must be in the front
is_deduced(And(Implies(And(is_playing(H), team_of(H) == team2), And(is_playing(Q), team_of(Q) == team1, position_of(Q) == middle)), is_playing(G), position_of(G) == front, team_of(G) == team1), team_of(H) == team2) ::: (7) H must be in team 2
'''


logic_program_3 = '''
# Declarations
magicians = EnumSort([G, H, K, L, N, P, Q]) 
positions = EnumSort([front, middle, back, no_position]) 
teams = EnumSort([team1, team2, no_team])
team_of = Function([magicians] -> [teams]) 
position_of = Function([magicians] -> [positions]) 
is_playing = Function([magicians] -> [bool]) 

# Constraints
Count([m:magicians], is_playing(m)) == 6 ::: Choose 6 people to play
And(Count([m:magicians], team_of(m) == team1) == 3, Count([m:magicians], team_of(m) == team2) == 3) ::: # Each team has exactly 3 magicians
And(Count([m:magicians], And(team_of(m) == team1, position_of(m) == front)) == 1, Count([m:magicians], And(team_of(m) == team1, position_of(m) == middle)) == 1, Count([m:magicians], And(team_of(m) == team1, position_of(m) == back)) == 1, Count([m:magicians], And(team_of(m) == team2, position_of(m) == front)) == 1, Count([m:magicians], And(team_of(m) == team2, position_of(m) == middle)) == 1, Count([m:magicians], And(team_of(m) == team2, position_of(m) == back)) == 1) ::: Each position in each team is occupied by exactly one magician
ForAll([m:magicians], is_playing(m) == And(team_of(m) != no_team, position_of(m) != no_position)) ::: A magician is playing if and only if they are assigned to a team and a position
ForAll([m:magicians], Not(is_playing(m)) == And(team_of(m) == no_team, position_of(m) == no_position)) ::: If a magician is not playing, they are assigned to no team and no position
And(Implies(is_playing(G), position_of(G) == front), Implies(is_playing(H), position_of(H) == front)) ::: (1) If G or H is arranged to play, they must be in the front
Implies(is_playing(K), position_of(K) == middle) ::: (2) If K is scheduled to play, he must be in the middle
Implies(is_playing(L), team_of(L) == team1) ::: (3) If L is scheduled to play, he must be on team 1
And(Implies(And(is_playing(P), is_playing(N)), team_of(P) != team_of(N)), Implies(And(is_playing(K), is_playing(N)), team_of(K) != team_of(N))) ::: (4) Neither P nor K can be in the same team as N
Implies(And(is_playing(P), is_playing(Q)), team_of(P) != team_of(Q)) ::: (5) P cannot be in the same team as Q
Implies(And(is_playing(H), team_of(H) == team2), And(is_playing(Q), team_of(Q) == team1, position_of(Q) == middle)) ::: (6) If H is in team 2, Q is in the middle of team 1

# Verifications
is_deduced(And(Implies(is_playing(G), position_of(G) == front), Implies(is_playing(H), position_of(H) == front), Implies(is_playing(L), team_of(L) == team1), is_playing(L), position_of(L) != front), Or(position_of(L) == middle, position_of(L) == back)) ::: (1) he must be in the middle or back
is_deduced(And(Implies(is_playing(K), position_of(K) == middle), is_playing(L), position_of(L) == middle), position_of(K) == back) ::: (2) K must be in the back ::: (2) K must be in the back
is_deduced(And(Implies(And(is_playing(P), is_playing(N)), team_of(P) != team_of(N)), Implies(And(is_playing(K), is_playing(N)), team_of(K) != team_of(N)), is_playing(K), position_of(K) == back), Not(position_of(P) == back)) ::: (3) P cannot be in the back
is_deduced(And(Implies(And(is_playing(P), is_playing(N)), team_of(P) != team_of(N)), Implies(And(is_playing(K), is_playing(N)), team_of(K) != team_of(N)), is_playing(K), position_of(K) == back, is_playing(P)), Or(position_of(P) == front, position_of(P) == middle)) ::: (4) P must be in the front or middle
is_deduced(Implies(And(is_playing(P), is_playing(Q)), team_of(P) != team_of(Q)), Implies(And(is_playing(P), is_playing(L)), team_of(P) != team_of(L))) ::: (5) P cannot be in the same team as L
is_deduced(And(Implies(And(is_playing(P), is_playing(N)), team_of(P) != team_of(N)), Implies(And(is_playing(K), is_playing(N)), team_of(K) != team_of(N)), is_playing(K), position_of(K) == back, is_playing(P), Implies(And(is_playing(P), is_playing(Q)), team_of(P) != team_of(Q)), team_of(L) == team_of(Q)), position_of(P) == front) ::: (6) P must be in the front
is_deduced(And(Implies(And(is_playing(H), team_of(H) == team2), And(is_playing(Q), team_of(Q) == team1, position_of(Q) == middle)), is_playing(G), position_of(G) == front, team_of(G) == team1), team_of(H) == team2) ::: (7) H must be in team 2
'''

logic_program_4 = '''
# Declarations
porcelains = EnumSort([S, Y, M, Q, K, X]) ::: haha(G)
positions = EnumSort([1, 2, 3, 4, 5, 6]) ::: (Permutation constraint: each porcelain has a distinct position)
position_of = Function([porcelains] -> [positions])
ForAll([p:porcelains], And(1 <= position_of(p), position_of(p) <= 6))

# Constraints
Distinct([p:porcelains], position_of(p)) ::: (Permutation constraint: each porcelain has a distinct position)
ForAll([p:porcelains], And(1 <= position_of(p), position_of(p) <= 6)) ::: (Position range constraint)
position_of(M) < position_of(X) ::: (1) M is older than X
Implies(position_of(Y) < position_of(M), And(position_of(Q) < position_of(K), position_of(Q) < position_of(X))) ::: (2) If Y is earlier than M, then Q is earlier than K and X
Implies(position_of(M) < position_of(Y), And(position_of(K) < position_of(Q), position_of(K) < position_of(X))) ::: (3) If the age of M is earlier than Y, the age of K is earlier than Q and X
Or(position_of(S) < position_of(Y), position_of(S) < position_of(M)) ::: (4) The age of S is either earlier than Y or earlier than M
Not(And(position_of(S) < position_of(Y), position_of(S) < position_of(M))) ::: (4) and both have neither

# Verifications
is_deduced(position_of(Y) < position_of(M), And(position_of(Y) < position_of(M), position_of(Q) < position_of(K), position_of(Q) < position_of(X))) ::: (1) if Y is earlier than M, the order is Y, M, Q, K, X
is_deduced(position_of(M) < position_of(Y), And(position_of(M) < position_of(Y), position_of(K) < position_of(Q), position_of(K) < position_of(X))) ::: (2) if M is earlier than Y, the order is M, K, Q, X
is_deduced(Or(position_of(S) < position_of(Y), position_of(S) < position_of(M)), Not(And(position_of(Y) < position_of(S), position_of(S) < position_of(M)))) ::: (3) S can be placed anywhere in the sequence, but it cannot be placed between Y and M
'''

logic_program_5 = '''
# Declarations
porcelains = EnumSort([S, Y, M, Q, K, X])
position_of = Function([porcelains] -> [int])
ForAll([p:porcelains], And(1 <= position_of(p), position_of(p) <= 6))

# Constraints
Distinct([p:porcelains], position_of(p)) ::: (Permutation constraint: each porcelain has a distinct position)
ForAll([p:porcelains], And(1 <= position_of(p), position_of(p) <= 6)) ::: (Position range constraint)
position_of(M) < position_of(X) ::: (1) M is older than X
Implies(position_of(Y) < position_of(M), And(position_of(Q) < position_of(K), position_of(Q) < position_of(X))) ::: (2) If Y is earlier than M, then Q is earlier than K and X
Implies(position_of(M) < position_of(Y), And(position_of(K) < position_of(Q), position_of(K) < position_of(X))) ::: (3) If the age of M is earlier than Y, the age of K is earlier than Q and X
Or(position_of(S) < position_of(Y), position_of(S) < position_of(M)) ::: (4) The age of S is either earlier than Y or earlier than M
Not(And(position_of(S) < position_of(Y), position_of(S) < position_of(M))) ::: (4) and both have neither

# Verifications
is_deduced(position_of(Y) < position_of(M), And(position_of(Y) < position_of(M), position_of(Q) < position_of(K), position_of(Q) < position_of(X))) ::: (1) if Y is earlier than M, the order is Y, M, Q, K, X
is_deduced(position_of(M) < position_of(Y), And(position_of(M) < position_of(Y), position_of(K) < position_of(Q), position_of(K) < position_of(X))) ::: (2) if M is earlier than Y, the order is M, K, Q, X
is_deduced(Or(position_of(S) < position_of(Y), position_of(S) < position_of(M)), Not(And(position_of(Y) < position_of(S), position_of(S) < position_of(M)))) ::: (3) S can be placed anywhere in the sequence, but it cannot be placed between Y and M
'''

logic_program_6 = '''
# Declarations
students = EnumSort([A, B, C, D])  
universities = EnumSort([Peking, Tsinghua, Nanjing, Southeast])  
attends = Function([students] -> [universities])  

# Constraints
attends(A) != Peking ::: (1) A did not attend Peking University  
attends(B) != Tsinghua ::: (2) B did not attend Tsinghua University  
attends(C) != Nanjing ::: (3) C did not attend Nanjing University  
attends(D) != Southeast ::: (4) D did not attend Southeast University  
Distinct([ s : students], attends(s)) ::: (Unique assignment: each student attends a different university)  
attends(A) == Tsinghua ::: (Derived from cot_parsing: A must have attended Tsinghua University)  
attends(B) == Peking ::: (Derived from cot_parsing: B must have attended Peking University)  
Or(attends(C) == Peking, attends(C) == Southeast) ::: (Derived from cot_parsing: C must have attended either Peking University or Southeast University)  
Or(attends(D) == Tsinghua, attends(D) == Nanjing) ::: (Derived from cot_parsing: D must have attended either Tsinghua University or Nanjing University)  

# Verifications
is_deduced(And(attends(A) == Tsinghua, attends(B) == Peking), True) ::: (1) It is possible that A attended Tsinghua University and B attended Peking University  
is_deduced(Or(attends(C) == Peking, attends(C) == Southeast), True) ::: (2) C must have attended either Peking University or Southeast University  
is_deduced(Or(attends(D) == Tsinghua, attends(D) == Nanjing), True) ::: (3) D must have attended either Tsinghua University or Nanjing University  
is_deduced(attends(A) == Tsinghua, True) ::: (4) A must have attended Tsinghua University  
is_deduced(attends(B) == Peking, True) ::: (5) B must have attended Peking University  
is_deduced(attends(D) != Peking, True) ::: (6) D did not attend Peking University
'''

logic_program_7 = '''
# Declarations
artists = EnumSort([A, B, C, D])  
fields = EnumSort([dancer, painter, singer, writer])  
field_of = Function([artists] -> [fields])

# Constraints
field_of(A) != singer ::: (1) A and C are not the singer  
field_of(C) != singer ::: (1) A and C are not the singer  
Not(Or(field_of(A) == painter, field_of(C) == painter)) ::: (2) The artist who painted for B and the writer is not A or C  
field_of(A) == writer ::: (3) A is a writer  
field_of(B) == singer ::: (4) B is the singer  
field_of(C) == painter ::: (5) C is the painter  
Distinct([A, B, C, D], [field_of(A), field_of(B), field_of(C), field_of(D)]) ::: (Uniqueness constraint)

# Verifications
is_deduced(And(field_of(A) != singer, field_of(C) != singer), True) ::: (1) A and C are not the singer  
is_deduced(Not(Or(field_of(A) == painter, field_of(C) == painter)), True) ::: (2) The artist who painted for B and the writer is not A or C  
is_deduced(field_of(A) == writer, True) ::: (3) A is a writer  
is_deduced(field_of(B) == singer, True) ::: (4) B is the singer  
is_deduced(field_of(C) == painter, True) ::: (5) C is the painter
'''


z3_program = LLMSR_Z3_Program(logic_program_2)
print(z3_program.standard_code)
output, error_message = z3_program.execute_program()
if error_message:
    print(f"Error: {error_message}")
else:
    print(z3_program.standard_code)
    print("Verification results:")
    results = extract_verification_results(output)
    print(results)

