from itertools import combinations
from math import gcd

def normalize_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    # 一般式: A*x + B*y + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    # 约分，保证唯一性
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g != 0:
        A //= g
        B //= g
        C //= g

    # 固定符号，避免 (A,B,C) 和 (-A,-B,-C) 被当成两条线
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C

    return (A, B, C)

def count_unique_lines(s):
    nums = list(map(int, s.split(',')))
    if len(nums) % 2 != 0:
        raise ValueError("坐标数量不是偶数，无法组成 (x,y) 点对。")

    points = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]

    # 如果有重复点，先去重；重复点不会增加新直线
    points = list(set(points))

    lines = set()
    for p1, p2 in combinations(points, 2):
        if p1 != p2:
            lines.add(normalize_line(p1, p2))

    return len(lines)

# 你的数据
s = "-6,-6,-5,1,-3,0,5,6,-3,6,-3,0,6,-4,-6,6,-2,-6,-5,-6,-5,5,-6,1,1,1,1,-5,4,-6,-2,-4,-2,3,-5,6,0,-5,3,3,-2,-2,4,4,5,-2,2,-2,-1,1,-3,-1,-3,-6,-5,-6,-5,-2,6,3,3,-5,-4,2,2,-6,-3,2,-5,4,-2,-3,-1,6,4,1,5,2,2,5,2,-3,-3,1,-3,-6,-1,-4,4,2,-3,-1,-4,3,0,-4,4,5,3,2,-5,-1,5,0,4,-5,6,0,6,-2,3,-3,-1,-6,-5,3,-5,-3,0,1,-6,-2,1,-5,-6,6,0,2,-3,4,-2,4,2,6,-2,6,4,5,-1,2,6,-3,-6,-2,-1,-3,-6,3,2,5,0,0,0,-4,3,5,-6,-4,0,4,3,-5,0,-3,-5,6,0,-2,-1,6,2,1,-1,-6,-5,-6,5,-2,0,-6,-4,-4,3,5,-1,-5,1,1,2,0,3,1,-1,3,-5,-5,6,4,4,-3"

print("不重复直线条数:", count_unique_lines(s))