from numpy.random import rand, choice

wallExistsInRow = 0.4
twoWallsExistsInRow = 0.2
patternLength = 23

nxt = {0: [1], 1: [0, 2], 2: [1]}

def dfs(row, pos, grid):
    if row == patternLength:
        return True
    if grid[row][pos] == 1:
        return False
    else:
        res = []
        if dfs(row+1, pos, grid):
            return True
        for p in nxt[pos]:
            if not grid[row][p]:
                if dfs(row+1, p, grid):
                    return True
        return False

def generatePattern():
    res = []
    for i in range(patternLength):
        row = [0, 0, 0]
        if rand() <= twoWallsExistsInRow:
            for ind in choice([0, 1, 2], size=2):
                row[ind] = 1
        else:
            if rand() <= wallExistsInRow:
                for ind in choice([0, 1, 2], size=1):
                    row[ind] = 1
        res.append(row)
    return res

if __name__ == '__main__':
    # pattern = generatePattern()
    # i = 0
    # while not dfs(0, 1, pattern):
    #     print(i)
    #     pattern = generatePattern()
    #     i += 1
    # print(pattern)

    with open('patterns.txt', 'w') as file:
        for i in range(80):
            pattern = [[0, 0, 0]]
            pat = generatePattern()
            while not dfs(0, 1, pat):
                pat = generatePattern()
            pattern += pat
            pattern.append([0, 0, 0])
            file.write(str(pattern) + '\n')