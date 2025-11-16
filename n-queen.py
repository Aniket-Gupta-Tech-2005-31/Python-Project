N = 4

def printBoard(b):
    for r in b:
        print(r)

def isSafe(b, r, c):
    for i in range(c):
        if b[r][i] == 1: return False
    for i, j in zip(range(r, -1, -1), range(c, -1, -1)):
        if b[i][j] == 1: return False
    for i, j in zip(range(r, N), range(c, -1, -1)):
        if b[i][j] == 1: return False
    return True

def solve(b, c=0):
    if c >= N: return True
    for r in range(N):
        if isSafe(b, r, c):
            b[r][c] = 1
            if solve(b, c+1): return True
            b[r][c] = 0
    return False

b = [[0]*N for _ in range(N)]
if solve(b): printBoard(b)
else: print("No solution")
