# Q1 3 month interview preparation
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'plusMinus' function below.
#
# The function accepts INTEGER_ARRAY arr as parameter.
#

def plusMinus(arr):
    # Write your code here
    total_len = len(arr)
    positive, negative, zero = [],[],[]
    for i in arr:
        if i > 0:
            positive.append(i)
        elif i < 0:
            negative.append(i)
        else:
            zero.append(i)
    positive_ratio,negative_ratio, zero_ratio = len(positive)/total_len,len(negative)/total_len,len(zero)/total_len
    print(format(positive_ratio,'.6f'), '\n',format(negative_ratio,'.6f'),'\n',format(zero_ratio,'.6f'),'\n')
        

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    plusMinus(arr)

# Q2
def miniMaxSum(arr):
    # Write your code here
    arr.sort()
    min_sum = sum(arr[:4])
    max_sum = sum(arr[::-1][:4])
    print(min_sum,max_sum)

# Q3
def timeConversion(s):
    # Write your code here
    # 12 hour to 24 hour
    # if AM: 11AM = 11, 12AM = 0, 9AM = AM, no 13AM
    # if PM: 1PM = 13
    if s[-2:] == 'AM':
        if int(s.split(":")[0]) == 12:
            return '00' + s[2:-2]
        else:
             return s[:-2]
    elif s[-2:] == 'PM':
        if int(s.split(":")[0]) == 12:
            return s[:-2]
        # 11pm --> 11+12 --> 23
        else:
            return str((int(s.split(":")[0])) + 12) + s[2:-2]
    # if int(s[:2]) == 12:
    #     if s[-2:] == 'AM':
    #         return str(int(s[:2]) - 12) + s[2:-2]
    #     else:
    #         return s[:-2]
    # if int(s[:2]) > 12:
    #     return str(int(s[:2]) - 12) + s[2:-2]
    # else:
    #     return s[:-2]
    # if s[::-1][:2] == "AM":
    #     return str(int(s[:2]) - 12) + s[2:-2]
    # else:
    #     return str(int(s[:2]) + 12) + s[2:-2]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = timeConversion(s)

    fptr.write(result + '\n')

    fptr.close()

# Q4
# Q5
# Complete the 'divisibleSumPairs' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER k
#  3. INTEGER_ARRAY ar
#

def divisibleSumPairs(n, k, ar):
    # Write your code here
    sum_ct = []
    for i in range(n-1):
        for j in range(i+1,n):
            if (ar[i] + ar[j])% k == 0:
                sum_ct.append((ar[i] + ar[j]))
    return len(sum_ct) 

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = int(first_multiple_input[0])

    k = int(first_multiple_input[1])

    ar = list(map(int, input().rstrip().split()))

    result = divisibleSumPairs(n, k, ar)

    fptr.write(str(result) + '\n')

    fptr.close()

    # Q6
def matchingStrings(strings, queries):
    # Write your code here
    output = []
    for i in queries:
        output.append(strings.count(i))
    return output
    # Dictionary doesn't work here because the keys can't be duplicates
    # counter = {}
    # for i in queries:
    #     for j in strings:
    #         if i == j:
    #             if i not in counter:
    #                 counter[i] = 1
    #             else:
    #                 counter[i] += 1
    #     if i not in strings:
    #         counter[i] = 0
    # return list(counter.values())
            

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    strings_count = int(input().strip())

    strings = []

    for _ in range(strings_count):
        strings_item = input()
        strings.append(strings_item)

    queries_count = int(input().strip())

    queries = []

    for _ in range(queries_count):
        queries_item = input()
        queries.append(queries_item)

    res = matchingStrings(strings, queries)

    fptr.write('\n'.join(map(str, res)))
    fptr.write('\n')

    fptr.close()
    
# Wk2 Q1 eaz
def lonelyinteger(a):
    # Write your code here
    temp = list(set(a))
    for i in temp:
        if a.count(i) == 1:
            return i

        # Wk2 Q2 brain not working
def gradingStudents(grades):
    # Write your code here
    #multiple_5 = [i for i in range(40,101) if i%5 == 0]
    output_grade = []
    for i in (grades):
        if i < 38:
            output_grade.append(i)
        else:
            temp = [k for k in range(i,101) if k % 5 == 0]
            diff = temp[0] - i
            #print(diff)
            #print([i,temp[0]])
#2
# [73, 75]
# 3
# [67, 70]
            if diff < 3:
                output_grade.append(temp[0])
            else:
                output_grade.append(i)
            # if 0 < diff < 3:
            #     output_grade.append(temp[0])
            # output_grade.append(i)
    grades = output_grade
    return grades
            
    # if grades < 40:
    #     return grades
    # for i in multiple_5:
    #     if 0 <= i - grades < 3:
    #         return i-grades
    # return grades 
    
    
# Q3
# Q4
def diagonalDifference(arr):
    # Write your code here
    # squre matrix: row == column
    right,left = [],[]
    row = len(arr[0])
    col = len(arr)
    for i in range(len(arr)):
        col += 1
        for j in range(len(arr[i])):
            if i == j:
                right.append(arr[i][j])
    abs_right = sum(right)
    # 0 2; 1;1; 2 0 
    for i in range(row):
        # 0 4; 1 3; 2 2; 3 1; 4 0
        # [0,3],[1,2]
        #print([i,row-i])
        left.append(arr[i][row-i-1])
    abs_left = sum(left)
    return abs(abs_right-abs_left)

# Q5
