from numpy.random import rand, choice

obstacleExistsInRow = 0.3
twoObstaclesExistsInRow = 0.2
obstacleIsDitch = 1/3
obstacleOverhead = 10/30
patternLength = 10
obstacleCount = int(patternLength*0.3)

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
        if rand() <= twoObstaclesExistsInRow:
            for ind in choice([0, 1, 2], size=2):
                row[ind] = 2 if rand() <= obstacleIsDitch else 1
        else:
            if rand() <= obstacleExistsInRow:
                for ind in choice([0, 1, 2], size=1):
                    row[ind] = 2 if rand() <= obstacleIsDitch else 1
        if rand() <= obstacleOverhead:
            for ind in choice([0, 1, 2], size=3):
                if row[ind] != 2 and row[ind] != 1:
                    row[ind] = 3
                    break
        res.append(row)
    counter = obstacleCount
    for row in res:
        for col in row:
            if col != 0 and col != 3: 
                counter -= 1
    if counter > 0:
        res = generatePattern()

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