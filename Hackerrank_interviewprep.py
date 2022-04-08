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
