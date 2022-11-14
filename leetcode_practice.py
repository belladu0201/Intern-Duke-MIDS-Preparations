# 121. Best Time to Buy and Sell Stock
# Jeppy teaches me this and we quarrel for this question lol need to remember this question then
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        cur_min = prices[0]
        diff_max = 0
        for i in prices:
            if i <= cur_min:
                cur_min = i
            diff = i - cur_min
            diff_max = max(diff_max,diff)
        return diff_max
 # second time writing this question, need to remember this algorithm  
def maxProfit(self, prices: List[int]) -> int:
        min_val = prices[0]
        diff = []
        for i in prices:
            cur = i
            min_val = min(min_val,i)
            diff.append(cur-min_val)
        return max(diff)

# 217. Contains Duplicate
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums) != len(set(nums))
    
# 238. Product of Array Except Self
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        temp,product = [],1
        zero = nums.count(0)
        for i in nums:
            if zero == 0:
                product *= i
            else:
                if i != 0:
                    product *= i
        for i,e in enumerate(nums):
            if zero == 0:
                temp.append(int(product/e))
            else:
                if (zero > 1):
                    temp.append(0)
                elif (zero == 1):
                    if e == 0:
                        temp.append(int(product))
                    else:
                        temp.append(0)
        return temp

    # 53. Maximum Subarray
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        max_cur,max_val = nums[0],nums[0]
        for i in range(0,len(nums)-1):
            max_cur = max(nums[i+1],max_cur+nums[i+1])
            max_val = max(max_val,max_cur)
        return max_val
            
  # 1688. Count of Matches in Tournament     
# didn't figure it out immediately, the team not played will move on to semi-final
class Solution:
    def numberOfMatches(self, n: int) -> int:
        sum_ = 0
        if n < 2:
            return 0
        while n >= 2:
            if n  % 2 == 0:
                n  = n / 2
                sum_ += n
            else:
                n = (n-1)/2 + 1
                sum_ += n-1
        return int(sum_)
        
# return n-1 using induction will prove this
# Base case : n = 2; n -> 1 --> base case matches
# Inductive step:
# n - 1 = n when n is an odd number
# thus n-1 = n = (n-1)/2 + (n-1)/4 + (n-1)/8 + ... + 1 --> n - 1
# (n-1)*(1/2 + 1/4 + 1/8 + ... + 1) = n - 1
# (1/2 + 1/4 + 1/8 + ... + 1) --> 1
# n + 1 = (n+1)*(1/2 + 1/4 + 1/8 + ... + 1) = n + 1 = n + 2

# 1. Two Sum
# Re-do the question. Last year I only know how to do it using nested for loops
# Now I learned that using hash map (dictionary) will make it with better time/space complexity
# I still reference the code from the discussion session, link can't be found anymore, thanks to the guy uploaded his/her answer
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashed = {}
        for idx, val in enumerate(nums):
            temp = target - val
            if temp not in hashed:
                hashed[val] = idx
            else:
                return [idx,hashed[temp]]
#2235. Add Two Integers
class Solution(object):
    def sum(self, num1, num2):
        """
        :type num1: int
        :type num2: int
        :rtype: int
        """
        return num1+num2
        
# 1929. Concatenation of Array
class Solution(object):
    def getConcatenation(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        return nums*2 #(or nums +nums)
        output = [i for i in nums]
        return output+output
#1920. Build Array from Permutation
class Solution(object):
    def buildArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        return [nums[i] for i in nums]
        
#2236. Root Equals Sum of Children
class Solution(object):
    def checkTree(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        return root.val == root.left.val + root.right.val
    
#1480. Running Sum of 1d Array
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        temp = nums.copy()
        for idx, val in enumerate(nums):
            nums[idx] = sum(temp[:idx+1])
        return nums
#  2114. Maximum Number of Words Found in Sentences
class Solution:
    def mostWordsFound(self, sentences: List[str]) -> int:
        temp = {}
        for i in sentences:
            temp[i] = len(i.split(" "))
        return max(temp.values())
        
# 2011. Final Value of Variable After Performing Operations
class Solution:
    def finalValueAfterOperations(self, operations: List[str]) -> int:
        output = 0
        for i in operations:
            if i in ['++X', 'X++']:
                output += 1
            elif i in ['--X', 'X--']:
                output -= 1
        return output
 # 1470. Shuffle the Array
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        front = nums[:n]
        end =nums[n:]
        temp = []
        for i in range(n):
            temp.append(front[i])
            temp.append(end[i])
        return temp
        
#1512. Number of Good Pairs
class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums)-1):
            temp = nums[i]
            if temp in nums[i+1:]:
                count += nums[i+1:].count(temp)
        return count
# 2160. Minimum Sum of Four Digit Number After Splitting Digits
class Solution:
    def minimumSum(self, num: int) -> int:
        # 1 3; 2 2; 3 1
        # the above is the only three possibilities
        num = [int(i) for i in str(num)]
        num.sort()
        temp1 = [int(str(num[0]) + str(num[2]))]
        temp2 = [int(str(num[1]) + str(num[3]))]
        return sum(temp1+temp2)
  # find out another better way to do it: 10* (num[0] + num[1]) + num[2]+ num[3]
# since we want the min sum, so we just take the min 2 numbers as the tenth digit number and let the larger two be the next digit

# 1588. Sum of All Odd Length Subarrays       
class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        # want all the odd sums
        output = 0
        for i in range(len(arr)):
            for j in range(i,len(arr)):
                # [0,1], [1,2],[2,3],[3,4],[4,5]
                # [0,3], [2,5]
                # [0,5]
                if (i+j) % 2 == 0:
                    output+=sum(arr[i:j+1])
        return output

# 70. Climbing Stairs
class Solution:
    def climbStairs(self, n: int) -> int:
        # n = 4
        # 1 1 1 1; 1 2 1; 1 1 2; 2 1 1; 2 2
        # 4-->5; 1--> 1; 2-->2; 3--> 3
        # 1 2 3 5...
        if n <= 2:
            return n
        start = [1,2]
        for i in range(2,n):
            start.append(sum(start[-2:]))
        return start[-1]
# 283. Move Zeroes
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        # in-place means that I can't create a new list
        zero_count=nums.count(0)
        for i in range(zero_count):
            nums.remove(0)
            nums.append(0)
        #for i in range(zero_count):
            #nums.append(0)
        return nums
# 94. Binary Tree Inorder Traversal               
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        def inorder_traversal(node):
            if node == None:
                return []
            return inorder_traversal(node.left) + [node.val] + inorder_traversal(node.right)
        return inorder_traversal(root)        
# 206 Reverse Linked List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        curr = head
        prev = None
        while curr != None:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev

    # 125. Valid Palindrome
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        temp = ""
        for i in s:
            #print(i)
            if i.isalnum():
                temp += i
                #print(i)
        temp = temp.lower()
        #print(temp)
        pt1,pt2 = 0,len(temp)-1
        #if len(temp) == 1: return False
        while pt1 < pt2:
            if temp[pt1] != temp[pt2]:
                return False
            else:
                pt1 +=1
                pt2 -=1
                #print("runned")
        return True
        
# 344. Reverse String
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        length = len(s)
        # if length  % 2 == 0:
        # use two pointers again
        left,right  = 0,length-1
        while left < right:
            s[left],s[right] = s[right],s[left]
            left += 1
            right -=1
        return s
        # for i in range(0,length % 2+1):
        #     s[i],s[-(i+1)] = s[-(i+1)],s[i]
        # return s

# 905. Sort Array By Parity
class Solution(object):
    def sortArrayByParity(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        return ([i for i in nums if i % 2 == 0] + [i for i in nums if i % 2 != 0])
        
# 1480. Running Sum of 1d Array
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        temp = []
        for i in range(len(nums)):
            temp.append(sum(nums[:i+1]))
        return temp
# Revised solution based on discussion
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        # temp = []
        # for i in range(len(nums)):
        #     temp.append(sum(nums[:i+1]))
        # return temp
        for i in range(1,len(nums)):
            nums[i] += nums[i-1]
        return nums
  
# 724. Find Pivot Index
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        # if sum(nums) == 0:
        #     return 0
        
        
        # time exceeds
        # for i in range(len(nums)):
        #     #print([sum(nums[:i]),sum(nums[i:])])
        #     if sum(nums[:i]) == sum(nums[i+1:]):
        #         return i
        # return -1
        
        # using two pointers
        l1 = 0
        l2 = sum(nums)
        for i in range(len(nums)):
            l2 -= nums[i]
            if l1 == l2:
                return i
            l1 += nums[i]
            
        return -1
        
   # 412. Fizz Buzz
class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        #temp =  [str(i) for i in range(1,n+1)]
        output = []
        for i in range(1,n+1):
            if (i % 3 == 0) and (i % 5 == 0):
                output.append("FizzBuzz")
            elif i % 3 == 0:
                output.append("Fizz")
            elif i % 5 == 0:
                output.append("Buzz")
            else:
                output.append(str(i))
        return output
                
    
# 19 Binary Search        
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            pivot = left + (right - left) // 2
            if nums[pivot] == target:
                return pivot
            if target < nums[pivot]:
                right = pivot - 1
            else:
                left = pivot + 1
        return -1

    # 202. Happy Number
    class Solution:
        def isHappy(self, n: int) -> bool:
            def sos(n):
                temp = 0
                for i in range(len(str(n))):
                    temp += int(str(n)[i])**2
                return temp
            
            visit = set()
            while n not in visit:
                visit.add(n)
                n = sos(n)
                if n == 1:
                    return True
            return False
        
# 205. Isomorphic Strings
# mapping with dictionary
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        s_dict,t_dict = {},{}
        for s1,t1 in zip(s,t):
            if (s1 not in s_dict) and (t1 not in t_dict):
                s_dict[s1] = t1
                t_dict[t1] = s1
                
            elif (s_dict.get(s1) != t1) or (t_dict.get(t1) != s1):
                return False
            
        return True
    
# 392. Is Subsequence    
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        # dynamic programming and two pointers
        # Approach 1: Divide and Conquer with Greedy, need to use recursion
# class Solution:
#     def isSubsequence(self, s: str, t: str) -> bool:
#         LEFT_BOUND, RIGHT_BOUND = len(s), len(t)

#         def rec_isSubsequence(left_index, right_index):
#             # base cases
#             if left_index == LEFT_BOUND:
#                 return True
#             if right_index == RIGHT_BOUND:
#                 return False
#             # consume both strings or just the target string
#             if s[left_index] == t[right_index]:
#                 left_index += 1
#             right_index += 1

#             return rec_isSubsequence(left_index, right_index)

#         return rec_isSubsequence(0, 0)

        
        
########################################################################################
        
        # Approach 2: Two Pointers
        
        # the length of the substring and the original string
        source,target = len(s),len(t)
        # set the two pointer at 0 for both strings
        l1 = l2 = 0
        
        # While loop to loop through the two strings, make sure it doesn't exceed the length of the string
        while l1 < source and l2 < target:
            # if the element in then the left(source) string to check the next char
            if s[l1] == t[l2]:
                l1 += 1
            # if not found we need to check the next char in the target string
            l2 += 1
         
        # check if the loop finishes the whole subsequence
        return l1 == source
# 21. Merge Two Sorted Lists
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # left,right = list1.val, list2.val
        
        # if any of the linked list is blank, return the other one
        # base case
        if list1 is None:
            return list2
        elif list2 is None:
            return list1
        
        # compare the value of each linked list
        if list1.val < list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1,list2.next)
            return list2
# 876. Middle of the Linked List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        temp = [head]
        while (temp[-1].next != None):
            temp.append(temp[-1].next)
        return temp[len(temp)//2]
#################### big brain method to find the median in the linkedin list ############################
class Solution:
    def middleNode(self, head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
# 409. Longest Palindrome
class Solution:
    def longestPalindrome(self, s: str) -> int:
        temp = {}
        result = 0
        for i in s:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] += 1
        for k in temp.values():
            result += k // 2 * 2
            if result % 2 == 0 and k % 2 == 1:
                # once this if statements met, it will never reach again, odd + even is always odd
                result += 1
        return result   
# 589. N-ary Tree Preorder Traversal
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if root is None:
            return []
        
        def dfs(node):
            if not node: return None
            res.append(node.val)
            for node in node.children:
                dfs(node)
                

        res = []
        dfs(root)
        return res
# 1281. Subtract the Product and Sum of Digits of an Integer
class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        temp = [int(i) for i in str(n)]
        product = 1
        sum_ = 0
        for i in temp:
            product *= i
            sum_ += i
        return product - sum_
    
#1365. How Many Numbers Are Smaller Than the Current Number
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        temp = nums.copy()
        nums.sort()
        return [nums.index(i) for i in temp]
            
# 1342. Number of Steps to Reduce a Number to Zero
class Solution:
    def numberOfSteps(self, num: int) -> int:
        temp = 0
        while num > 0:
            if num % 2 == 0:
                num /= 2
            else:
                num -=1
            temp += 1
        return temp

# 104. Maximum Depth of Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        else:
            l1 = self.maxDepth(root.left)
            r1 = self.maxDepth(root.right)
            return max(l1,r1) + 1
# 160. Intersection of Two Linked Lists
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        p1 = headA
        p2 = headB
        # check if the node is same or not
        while p1 != p2:      
            # if node is None, need to change it to the other linked list
            p1 = headB if p1 is None else p1.next
            p2 = headA if p2 is None else p2.next           
        return p1
# 728. Self Dividing Numbers
class Solution(object):
    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """
        # using brute force
        temp = []
        for i in range(left,right + 1):
            for j in str(i):
                if int(j) == 0 or i % int(j)!=0:
                    break
            else:
                temp.append(i)
        return temp

# 1672. Richest Customer Wealth
class Solution(object):
    def maximumWealth(self, accounts):
        """
        :type accounts: List[List[int]]
        :rtype: int
        """
        return max([sum(i) for i in accounts])
# 1431. Kids With the Greatest Number of Candies
class Solution(object):
    def kidsWithCandies(self, candies, extraCandies):
        """
        :type candies: List[int]
        :type extraCandies: int
        :rtype: List[bool]
        """
        new = [i+extraCandies for i in candies]
        temp = []
        for i in range(len(new)):
            if new[i] >= max(candies):
                temp.append(True)
            else:
                temp.append(False)
        return temp
# 1313. Decompress Run-Length Encoded List
class Solution(object):
    def decompressRLElist(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        temp = []
        for i in range(0,len(nums),2):
            for j in range(nums[i]):
                temp.append(nums[i+1])
            
        return temp
# 832. Flipping an Image
class Solution(object):
    def flipAndInvertImage(self, image):
        """
        :type image: List[List[int]]
        :rtype: List[List[int]]
        """
        for i in range(len(image)):
            image[i] = image[i][::-1]
            image[i] = [0 if j == 1 else 1 for j in image[i]]
        return image
    
# 1572. Matrix Diagonal Sum
class Solution(object):
    def diagonalSum(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: int
        """
        # [3,3] --> [00,02,11,20,21]
        ct = 0
        for i in range(len(mat)):
            temp = len(mat) - i - 1
            for j in range(len(mat)):
                if j == temp:
                    print(mat[i][j])
                    ct += mat[i][j]
                elif j == i:
                    ct += mat[i][j]
        # ct += mat[0][0]
        # ct += (mat[len(mat)-1][len(mat)-1]) * (len(mat) - 2)
        return ct
# 1464. Maximum Product of Two Elements in an Array
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        temp = []
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i!= j:
                    temp.append((nums[i]-1)*(nums[j]-1))
        return max(temp)
    ##############################
        temp1 = max(nums)
        nums.remove(temp1)
        temp2 = max(nums)
        return (temp1-1)*(temp2-1)
    ######## MIN HEAP SOLUTION FROM DISCUSSION #########
    class Solution:
        def maxProduct(self, nums: List[int]) -> int:
            heap=[]
            for i in nums:
                heappush(heap,i)
                if len(heap)>2:
                    heappop(heap)
            return (heap[0]-1)*(heap[1]-1)
# 1266. Minimum Time Visiting All Points
class Solution(object):
    def minTimeToVisitAllPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        # temp = [points[0]]
        # steps = 0
        # for i in range(len(points)-1):
        #     while temp[i] != points[i+1][0] and temp[i] != points[i+1][1]:
        #         temp[i][0] -= 1
        #         temp[i][1] -= 1
        #         steps += 1
        #     # if temp[i] == points[i+1][0] or temp[i] == points[i+1][1]:
        #     #     steps += max(temp[0] - points[i][0], temp[1] - points[i][1])
        # return steps
                  
        # for i in range(len(points) - 1):
        #     if points[i][0] != points[i+1][0] and points[i][1] != points[i+1][1]:
        #         ct += min(abs(points[i+1][0]-points[i][0]),abs(points[i+1][1] - points[i][1]))
        #         ct += max(abs(points[i+1][0]-points[i][0]),abs(points[i+1][1] - points[i][1])) - 
        ct = 0  
        for i in range(len(points) - 1):
            ct += max(abs(points[i+1][0]-points[i][0]),abs(points[i+1][1] - points[i][1]))
        return ct
# 1304. Find N Unique Integers Sum up to Zero
class Solution(object):
    def sumZero(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        temp = []
        for i in range( 1, n // 2  + 1):
            temp.append(i)
            temp.append(-i)
        if n % 2 == 1:
            temp.append(0)
        return temp
# 509. Fibonacci Number
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 1 or n == 0: return n
        temp = [0,1]
        for i in range(n-1):
            temp.append(sum(temp[i:i+2]))
            
        return temp[-1]
# 1822. Sign of the Product of an Array
class Solution(object):
    def arraySign(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if 0 in nums: return 0
        if sum([1 for i in nums if i < 0]) % 2 == 0:
            return 1
        return -1
# 268. Missing Number
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return list(set([i for i in range(len(nums) + 1)]) - set(nums))[0]
# 1523. Count Odd Numbers in an Interval Range
class Solution(object):
    def countOdds(self, low, high):
        """
        :type low: int
        :type high: int
        :rtype: int
        """
        # ct = 0
        # for i in range(low,high+1):
        #     if i % 2 == 1:
        #         ct += 1
        # return ct
        # return sum([1 for i in range(low,high+1) if i % 2 == 1])
        
        ct = 0
        if low % 2 == 1 or high % 2 == 1:
            ct = 1
        return (high-low) // 2 + ct
# 1528. Shuffle String
class Solution(object):
    def restoreString(self, s, indices):
        """
        :type s: str
        :type indices: List[int]
        :rtype: str
        """
        # output = ""
        # for i in indices:
        #     output += s[i]
        # return output
        
        temp = [""] * len(indices)
        for i in range(len(indices)):
            temp[indices[i]] = s[i]
        return "".join(temp)
# 867. Transpose Matrix
class Solution(object):
    def transpose(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        temp = [[0 for j in range(len(matrix))] for i in range(len(list(zip(*matrix))))]
        for i in range(len(list(zip(*matrix)))):
            for j in range(len(matrix)):
                # print([i,j])
                temp[i][j] = matrix[j][i]
        print(len(matrix))
        print(len(list(zip(*matrix))))
        return temp
# 326. Power of Three
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        # if n == 1: return True
        if abs(n) < 1: return False
        while n % 3 == 0:
            n /= 3
        return n == 1
        
# 2176. Count Equal and Divisible Pairs in an Array
class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        ct = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] == nums[j] and i * j % k == 0:
                    ct += 1
        return ct
        # ct = 0
        # for i in range(len(nums)):
        #     if nums[i] not in temp:
        #         temp[nums[i]] = [i]
        #     else:
        #         temp[nums[i]].append(i)
        # for key,v in temp.items():
        #     if len(v) == 2:
        #         if (v[0]*v[1]) % k == 0:
        #             ct += 1
        # print(temp)
        # return ct
# 1086. High Five
class Solution:
    def highFive(self, items: List[List[int]]) -> List[List[int]]:
        temp = {}
        output = []
        for i in items:
            #print(i[0])
            if i[0] not in temp:
                temp[i[0]] = [i[1]]
            else:
                temp[i[0]].append(i[1])
        for k,v in temp.items():
            v.sort()
            v = v[::-1]
            output.append([k,sum(v[:5])//5])
        print(temp)
        output.sort()
        return output
# 1025. Divisor Game
class Solution:
    def divisorGame(self, n: int) -> bool:
        temp = n
        ct = 0
        while temp != 0:
            temp -=1
            ct +=1
        return ct % 2 == 0
# 62. Unique Paths
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
#         start = [0,0]
#         target = [m,n]
#         if start == target:
#             return 1
#         else:
#             return self.uniquePaths(start[0]+1, start[1]) + self.uniquePaths(start[0], start[1]+1)

        temp = [[1] * n for i in range(m)]
    
        for i in range(1,m):
            for j in range(1,n):
                temp[i][j] = temp[i][j-1] + temp[i-1][j]
        return temp[m-1][n-1]
# 383. Ransom Note
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        temp = {}
        mag = {}
        for i in ransomNote:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] += 1
        for i in magazine:
            if i not in mag:
                mag[i] = 1
            else:
                mag[i] += 1
        ct = 0
        for k,v in temp.items():
            if k in mag.keys() and v <= mag[k]:
                ct += 1
        return ct == len(temp)

    
# 1913. Maximum Product Difference Between Two Pairs
class Solution:
    def maxProductDifference(self, nums: List[int]) -> int:
        nums.sort()
        return (nums[-2] * nums[-1]) - (nums[1] * nums[0])

# 1732. Find the Highest Altitude
class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        start = [0]
        for i in gain:
            start.append(start[-1] + i)
        return max(start)
# 2089. Find Target Indices After Sorting Array
class Solution:
    def targetIndices(self, nums: List[int], target: int) -> List[int]:
        nums.sort()
        ct = nums.count(target)
        return [nums.index(target) + i for i in range(ct)]
        # if target in nums:
        #     return nums.index(target)
        # return targetIndices(nums[nums.index(target)+1:], target)
# 1351. Count Negative Numbers in a Sorted Matrix
class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        # brute force method
        ct = 0
        for i in grid:
            for j in i:
                if j < 0:
                    ct += 1
        return ct
# 557. Reverse Words in a String III
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.split(' ')
        temp = ""
        # l,r = 0,len(s)-1
        # while l <= r:
        #     temp[l],temp[r] = temp[r],temp[l]
        #     l += 1
        #     r -=1
        # return "".join(temp)
        for i in range(0,len(s)):
            temp += s[i][::-1]
            temp += " "
        return temp[:-1]
# ####### two pointer ############ #
class Solution:
    def reverseWords(self, s: str) -> str:
        #s = s.split(' ')
        temp = [i for i in s]
        l,r = 0,len(s)-1
        while l <= r:
            temp[l],temp[r] = temp[r],temp[l]
            l += 1
            r -=1
        curr = "".join(temp)
        return ' '.join(curr.split(" ")[::-1])
# 1332. Remove Palindromic Subsequences
class Solution:
    def removePalindromeSub(self, s: str) -> int:
        l,r = 0,len(s)-1
        s = [i for i in s]
        while l < r:
            if s[l] == s[r]:
                l += 1
                r -= 1
            else:
                return 2
        return 1
# 977. Squares of a Sorted Array
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        # temp = [i**2 for i in nums]
        # temp.sort()
        # return temp
        return sorted([i**2 for i in nums])
######## TWO POINTER SOLUTION ###########
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        # temp = [i**2 for i in nums]
        # temp.sort()
        # return temp
        
        # return sorted([i**2 for i in nums])
        
        result = [0] * len(nums)
        l,r = 0,len(nums) - 1
        for i in range(len(result)-1,-1,-1):
            if abs(nums[l]) > abs(nums[r]):
                curr = abs(nums[l])
                l +=1
            else:
                curr = abs(nums[r])
                r -=1
            result[i] = curr **2
        return result
# 1768. Merge Strings Alternately
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        l,r = len(word1), len(word2)
        result = ""
        for i in range(min(l,r)):
            result += word1[i] + word2[i]
        if l > r:
            for i in range(l-r):
                result += word1[min(l,r) + i]
        else:
            for i in range(r-l):
                result += word2[min(l,r) + i]
        return result
# 234. Palindrome Linked List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        temp = []
        while head.next != None:
            temp.append(head.val)
            head = head.next
        temp.append(head.val)
        return temp == temp[::-1]
            
# 680. Valid Palindrome II
class Solution:
    def validPalindrome(self, s: str) -> bool:
                    
        def recursive_palindrom(s,l,r):
            while l < r:
                if s[l] != s[r]:
                    return False
                else:
                    l += 1
                    r -= 1
            return True
        l,r = 0, len(s) - 1
        while l < r:
            if s[l] != s[r]:
                return recursive_palindrom(s,l,r-1) or recursive_palindrom(s,l+1,r)
            l +=1
            r -=1
        return True
# 1710. Maximum Units on a Truck
class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        boxTypes.sort(key=lambda x: -x[1])
        cur_size = 0
        max_units = 0
        for num_box, unit in boxTypes:
            max_units += unit * min(truckSize - cur_size, num_box)
            cur_size += min(truckSize - cur_size, num_box)
        return max_units
# 58. Length of Last Word
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        temp = s.split(" ")
        # result = []
        # for i in temp:
        #     for j in i:
        #         if j is 
        #         if j == " " or j == "":
        #             break
        #     print([i])
        #     result.append(i)
        # print(result)
        # return len(result[-1])
        ct = []
        temp1 = [i.isalpha() for i in temp]
        for i in range(len(temp1)):
            if temp1[i] == True:
                ct.append(i)
        
        return len(temp[ct[-1]])
    #     def lengthOfLastWord(self, s: str) -> int:
        temp = s.split(" ")
        return len(s.split().pop())
# 1876. Substrings of Size Three with Distinct Characters
class Solution:
    def countGoodSubstrings(self, s: str) -> int:
        l,r = 0,3
        ct = 0
        temp = []
        while r <= len(s):
            temp.append(s[l:r])
            l+=1
            r+=1
        for i in temp:
            if len(list(i)) == len(list(set(i))):
                ct += 1
        return ct
    #########class Solution:
    def countGoodSubstrings(self, s: str) -> int:
        count = 0
        for i in range(len(s)-2):
            if len(set([s[i],s[i+1],s[i+2]])) == 3:
                count+=1
        return count
# 2108. Find First Palindromic String in the Array
class Solution:
    def firstPalindrome(self, words: List[str]) -> str:
        def find_pali(i):
            l,r = 0,len(i)-1
            while l <= r:
                if i[l] != i[r]:
                    return False
                l += 1
                r -=1
            return True
        temp = []
        for i in words:
            temp.append(find_pali(i))
        if True in temp:
            return words[temp.index(True)]
        return ""
#             for word in words:
#             if word[0::] == word[::-1]:
#                 return word
#         return ""


# 2000. Reverse Prefix of Word
class Solution:
    def reversePrefix(self, word: str, ch: str) -> str:
        if ch not in word: return word
        idx = word.index(ch)
        word = [i for i in word]
        l,r = 0,idx
        while l <= r:
            word[l],word[r] = word[r],word[l]
            l += 1
            r -= 1
        return "".join(word)
# 349. Intersection of Two Arrays
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        temp = {}
        output  = []
        for i in nums1:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] += 1
        for i in nums2:
            if i in temp.keys():
                output.append(i)
        return list(set(output))
# 1380. Lucky Numbers in a Matrix
class Solution:
    def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
        candidates1 = [min(i) for i in matrix]
        candidates2 = []
        # for i in range(len(matrix)):
        #     temp = []
        #     for j in range(len(matrix[i])):
        #         temp.append(matrix[i][j])
        #     candidates2.append(max(temp))
        #print([len(matrix),len(matrix[0])]) row by col
        for i in range(len(matrix[0])):
            # loop through each column
            temp = []
            for j in range(len(matrix)):
                # loop through each row no.
                temp.append(matrix[j][i])
            candidates2.append(max(temp))
        #print([candidates1,candidates2])
        return list(set(candidates1) & set(candidates2))
# 492. Construct the Rectangle
class Solution:
    def constructRectangle(self, area: int) -> List[int]:
        L = W = int(math.sqrt(area))
        
        while L*W != area:
            if L*W < area:
                L += 1
            else:
                W -=1
        return [L,W]
# 231. Power of Two
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 1: return True
        if n <= 0: return False
        while n > 1:
            if int(n % 2) != 0:
                return False
            n /= 2
        return True
# 441. Arranging Coins
class Solution:
    def arrangeCoins(self, n: int) -> int:
        # 1 3 6 10 15 21
        ct = 0
        temp = 0
        while temp <= n:
            ct += 1
            temp += ct
        return ct-1
# 2206. Divide Array Into Equal Pairs
class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        temp = {}
        want = len(nums) // 2
        for i in nums:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] += 1
            if temp[i] == 2:
                temp[i] = 0
                want -=1
        return want == 0
# 1748. Sum of Unique Elements
class Solution:
    def sumOfUnique(self, nums: List[int]) -> int:
        temp = {}
        output = 0
        for i in nums:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] += 1
        for k,v in temp.items():
            if v == 1:
                output += k
        return output
        
# 1742. Maximum Number of Balls in a Box
class Solution:
    def countBalls(self, lowLimit: int, highLimit: int) -> int:
        ans = {}
        for i in range(lowLimit,highLimit+1):
            temp = sum([int(j) for j in str(i)])
            if temp not in ans:
                ans[temp] = 1
            else:
                ans[temp] += 1
        # for k,v in ans.items():
        #     if v == max(ans.values()):
        return max(ans.values())
# 2053. Kth Distinct String in an Array
class Solution:
    def kthDistinct(self, arr: List[str], k: int) -> str:
        temp = {}
        for i in arr:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] +=1
        ans = []
        for k1,v in temp.items():
            if v == 1:
                ans.append(k1)
        print(ans)
        if len(ans) == 0: return ""
        if len(ans) < k-1: return ""
        
        return ans[k-1]
# 2248. Intersection of Multiple Arrays
class Solution:
    def intersection(self, nums: List[List[int]]) -> List[int]:
        # if len(nums) <= 1:
        #     return []
        temp = set(nums[0])
        for i in range(1,len(nums)):
            temp = temp.intersection(nums[i])
        return sorted(temp)
        # def find_intersection(a,b):
        #     return list(set(a)) & list(set(b))
        # for i in range(len(nums)-1):
        #     find_intersection(nums[i],nums[i+1])
# 219. Contains Duplicate II
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        # hashmap
        temp = {}
        for idx,e in enumerate(nums):
            if e in temp and abs(idx-temp[e]) <= k:
                return True
            temp[e] = idx
            #print([idx,e])
        return False
        # Time Limit Exceed
        # for i in range(len(nums)-1):
        #     for j in range(i+1, i+2+k):
        #         if nums[i] == nums[j] and abs(i-j) <= k:
        #             return True
        # return False
# 83. Remove Duplicates from Sorted List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        temp = head
        lst = []
        while temp != None and temp.next != None:
            if temp.next.val == temp.val:
                temp.next = temp.next.next
            else:
                temp = temp.next
        return head
# 242. Valid Anagram
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        temp = {}
        result = {}
        for i in s:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] += 1
        for i in t:
            if i not in result:
                result[i] = 1
            else:
                result[i] += 1
        return temp == result
# 243. Shortest Word Distance
class Solution:
    def shortestDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        #temp = [abs(wordsDict.index(word1) - wordsDict.index(word2))]
        temp = []
        output = []
        for i in range(len(wordsDict)):
            if wordsDict[i] == word1 or wordsDict[i] == word2:
                temp.append(i)
        print(temp)
        for i in range(len(temp)-1):
            for j in range(i+1, len(temp)):
                if wordsDict[temp[i]] != wordsDict[temp[j]]:
                    #print([i,j])
                    output.append(abs(temp[i]-temp[j]))
        return min(output)
# 246. Strobogrammatic Number
class Solution:
    def isStrobogrammatic(self, num: str) -> bool:
        hashmap = {'0':'0', '1':'1', '8':'8', '6':'9', '9':'6'}
        temp = ""
        for i in num[::-1]:
            if i in hashmap:
                temp += hashmap[i]
        return temp == num
# 252. Meeting Rooms
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        # this is very similar to dsc20 or 30 homework questions
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i+1][0]:
                return False
        return True
# 263. Ugly Number
class Solution:
    def isUgly(self, n: int) -> bool:
        if n == 0: return False
        while n % 2 == 0:
            n /= 2
        while n % 3 == 0:
            n /= 3
        while n % 5 == 0:
            n /= 5
            
        return n == 1
# 266. Palindrome Permutation
class Solution:
    def canPermutePalindrome(self, s: str) -> bool:
        hmap = {}
        for i in s:
            if i not in hmap:
                hmap[i] = 1
            else:
                hmap[i] += 1
        ct = 0
        for v in hmap.values():
            if v % 2 !=0:
                ct += 1
        return ct == 1 or ct == 0
# 278. First Bad Version
# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        l,r = 1,n
        while (l < r):
            m = l + (r-l) // 2
            if isBadVersion(m):
                r = m
            else:
                l = m + 1
                
        return l
# 293. Flip Game
class Solution:
    def generatePossibleNextMoves(self, currentState: str) -> List[str]:
        output = []
        for i in range(len(currentState) - 1):
            temp = "".join([i for i in currentState])
            if currentState[i] == '+' and currentState[i+1]== '+': 
                temp = temp[:i] + temp[i].replace('+','-') + temp[i+1:]
                temp = temp[:i+1] + temp[i+1].replace('+','-') + temp[i+2:]
            #print(temp[i])
                output.append(temp)
        return output
# 108. Convert Sorted Array to Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def helper(left,right):
            if left > right:
                return None
        # Preorder
            m = (left + right) // 2
            root = TreeNode(nums[m])
            root.left = helper(left,m-1)
            root.right = helper(m+1,right)
            return root
        return helper(0,len(nums)-1)
#171. Excel Sheet Column Number
class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        temp = 0
        t = len(columnTitle)
        for i in range(t):
            temp = temp * 26
            temp += (ord(columnTitle[i]) - ord('A') + 1)
        return temp
        # num_digit = len([i for i in columnTitle])
        # #print(ord('A'))
        # temp = ord(columnTitle[-1]) - 64
        # #if num_digit == 1: return ord(columnTitle[-1]) - 64
        # for i in range(1,num_digit):
        #     temp += (i)*26 + ord(columnTitle[i]) - 64
        #     print([temp])
        # return temp
# 100. Same Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p == None and q == None:
            return True
        elif p == None or q == None:
            return False
        elif p.val != q.val:
            return False
        return self.isSameTree(p.right,q.right) and self.isSameTree(p.left,q.left)

# 1859. Sorting the Sentence
class Solution:
    def sortSentence(self, s: str) -> str:
        s = s.split(" ")
        temp = ["" for i in range(len(s))]
        print(s)
        for i in range(len(s)):
            #print(s[i][-1]) # 2 4 1 3
            temp[int(s[i][-1]) - 1] = s[i][:-1]
            #temp[s[i][-1] - 1] = s[i][:-1]
        return " ".join(temp)

# 1047. Remove All Adjacent Duplicates In String
class Solution:
    def removeDuplicates(self, s: str) -> str:
        # Use Stack to implement it
        stack = []
        for i in s:
            if stack and i == stack[-1]:
                stack.pop()
            else:
                stack.append(i)
        return ''.join(stack)

# 387. First Unique Character in a String
class Solution:
    def firstUniqChar(self, s: str) -> int:
        temp = {}
        for i in s:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] += 1
        for k,v in temp.items():
            if v == 1:
                return s.index(k)
        return -1


# 290. Word Pattern
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        s = s.split(" ")
        if len(pattern) != len(s): return False
        dic = {}
        for index in range(len(pattern)):
            if pattern[index] not in dic:
                if s[index] not in dic.values():
                    dic[pattern[index]] = s[index]
                else: return False
            else:
                if dic[pattern[index]] != s[index]:
                    return False
        return True

# 1002. Find Common Characters
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        temp = list(words[0])
        for i in words[1:]:
            new = []
            for c in i:
                if c in temp:
                    new.append(c)
                    temp.remove(c)
            temp = new
        return temp

# 345. Reverse Vowels of a String
class Solution:
    def reverseVowels(self, s: str) -> str:
        temp = []
        s = [i for i in s]
        for i in range(len(s)):
            if s[i] in ['a', 'e', 'i', 'o','u','A', 'E', 'I', 'O','U']:
                temp.append(i)
        # print(temp)
        for i in temp:
            if len(temp) >= 2:
                s[temp[0]],s[temp[-1]] = s[temp[-1]],s[temp[0]]
                temp = temp[1:-1]
            
        return ''.join(s)

# 697. Degree of an Array
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        temp = {}
        output = []
        for i in nums:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] += 1
        for k,v in temp.items():
            if v == max(temp.values()):
                output.append(k)
        print(output)
        final = []
        for j in output:
            store = []
            for i in range(len(nums)):
                if nums[i] == j:
                    store.append(i)
            print(store)
            final.append(store)
        num = []
        for i in final:
            num.append(i[-1] - i[0] + 1)
            
        return min(num)

# 566. Reshape the Matrix
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        # using queue to approach this question
        row = len(mat) # 2 
        col = len(mat[0]) # 4
        if r*c != row * col: return mat
        temp = []
        for i in range(row):
            for j in range(col):
                print(mat[i][j])
                temp.append(mat[i][j])
        # list comprehension
        return [[temp.pop(0) for i in range(c)] for j in range(r)]
        # output = []
        # ct = 0
        # if c >= r:
        #     for i in range(c//r):
        #         curr = []
        #         if ct < len(temp):
        #             for j in range(ct,ct+c//r):
        #                 curr.append(temp[j])
        #             ct += c//r
        #             output.append(curr)
        # else:
        #     for i in range(r):
        #         #print([i,i,i,i,i])
        #         curr = []
        #         if ct < len(temp):
        #             for j in range(ct,ct+j):
        #                 print([j])
        #                 curr.append(temp[j])
        #                 #print([curr,curr])
        #             ct += j
        #             output.append(curr)
        
        # ct = 0
        # for i in range(r):
        #     curr = []
        #     for j in range(ct+1,ct + c//r, c//r):
        #         curr.append(temp[j])
        #     output.append(curr)
        #     ct += c//r
        return output


# 917. Reverse Only Letters
class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        # using stack to solve problems
        temp = [i for i in s if i.isalpha()]
        ans = []
        for i in s:
            if i.isalpha():
                ans.append(temp.pop())
            else:
                ans.append(i)
        return "".join(ans)

# 944. Delete Columns to Make Sorted
class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        ct = 0
        for i in range(len(strs[0])):
            column = list(j[i] for j in strs)
            if column != sorted(column):
                ct += 1
        return ct
        # 001020 012131 022232 032333
        #curr = "".join([strs[i][j] for i in range(len(strs)) for j in  range(len(strs[0]))])
        #curr = "".join([k[i] for i in range(len(strs[0])) for k in strs])
        #curr = "".join([strs[i][j] for i in range(len(strs)) for j in range(len(strs[0]))])
        # for i in strs:
        #     if i != sorted(i):
        #         ct += 1
        # return ct
        # start = 0
        # end = len(strs[0])
        # while end <= len(curr):
        #     if curr[start:end] != "".join(sorted(curr[start:end])):
        #         print([curr[start:end],sorted(curr[start:end])])
        #         ct += 1
        #     start += len(strs[0])
        #     end += len(strs[0])
        # return ct

# Dot Product of Two Sparse Vectors
class SparseVector:
    def __init__(self, nums: List[int]):
        self.array = nums
        

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        left,right = 0,len(vec.array)-1
        add = 0
        while left <= right:
            add += self.array[left] * vec.array[left]
            left += 1
        return add
            
        
        

# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)

# 2161. Partition Array According to Given Pivot
class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        return [i for i in nums if i < pivot] + [i for i in nums if i == pivot] + [i for i in nums if i > pivot]
        # pos = nums.index(pivot)
        # l,r = 0,len(nums)-1

# 1877. Minimize Maximum Pair Sum in Array
class Solution:
    def minPairSum(self, nums: List[int]) -> int:
        nums.sort()
        pair_sum = 0
        max_idx = len(nums) - 1
        for i in range(len(nums)//2):
            pair_sum = max(pair_sum, nums[i] + nums[max_idx])
            max_idx -= 1
        return pair_sum
# 2149. Rearrange Array Elements by Sign
class Solution:
    def rearrangeArray(self, nums: List[int]) -> List[int]:
        p = [i for i in nums if i > 0]
        q = [i for i in nums if i < 0]
        l,r = 0, len(nums)//2 - 1
        ans = []
        while l <= r:
            ans.append(p[l])
            ans.append(q[l])
            l += 1
        return ans
# 1874. Minimize Product Sum of Two Arrays
class Solution:
    def minProductSum(self, nums1: List[int], nums2: List[int]) -> int:
        nums1.sort()
        nums2.sort()
        l,r = 0,len(nums1)-1
        ans = 0
        print([nums1,nums2])
        ct = 0
        while ct < (len(nums1)):
            ans += nums1[l] * nums2[r]
            #print(ans)
            l+=1
            r-=1
            ct += 1
        return ans
# 350. Intersection of Two Arrays II
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        ans = []
        hashset1,hashset2 = {},{}
        for i in nums1:
            if i not in hashset1:
                hashset1[i] = 1
            else:
                hashset1[i] += 1
        ############################
        for i in nums2:
            if i not in hashset2:
                hashset2[i] = 1
            else:
                hashset2[i] += 1
        ############################
        keys = hashset1.keys() & hashset2.keys() # find intersection
        for i in keys:
            ans += [i]*min(hashset1[i], hashset2[i]) 
        return ans
# 1382. Balance a Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        # we need to use in order tranversal
        ans = []
        # Hint 1: Convert the tree to a sorted array using an in-order traversal.
        def in_order(root):
            if not root: return None
            in_order(root.left)
            ans.append(root.val)
            in_order(root.right)
        in_order(root)
        # Hint 2: Construct a new balanced tree from the sorted array recursively.
        def check_balance(ans):
            if not ans: return None
            m = len(ans) // 2
            
            root = TreeNode(ans[m])
            root.left = check_balance(ans[:m])
            root.right = check_balance(ans[m+1:])
            return root
        return check_balance(ans)
# 2006. Count Number of Pairs With Absolute Difference K
class Solution:
    def countKDifference(self, nums: List[int], k: int) -> int:
        # use brute force to check each element
        ct = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if abs(nums[j]-nums[i]) == k:
                    ct += 1
        return ct

#  1662. Check If Two String Arrays are Equivalent
class Solution:
    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        temp1,temp2  = "",""
        for i in word1:
            temp1 += i
        for j in word2:
            temp2 += j
        return temp1 == temp2

# 485. Max Consecutive Ones
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        max_ct = ct= 0
        for i in nums:
            if i == 1:
                ct += 1
            else:
                max_ct = max(max_ct,ct)
                ct = 0
        return max(max_ct,ct)
# 1295. Find Numbers with Even Number of Digits
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        ct = 0
        for i in nums:
            if len([j for j in str(i)]) % 2 == 0:
                ct += 1
        return ct
# 2413. Smallest Even Multiple
class Solution:
    def smallestEvenMultiple(self, n: int) -> int:
        if n % 2 == 0:
            return n
        return n * 2
# 1389. Create Target Array in the Given Order
class Solution:
    def createTargetArray(self, nums: List[int], index: List[int]) -> List[int]:
        output = [] * len(nums)
        for i in range(len(nums)):
            output.insert(index[i], nums[i])
            
        return output

# 2180. Count Integers With Even Digit Sum
class Solution:
    def countEven(self, num: int) -> int:
        ct = 0
        for i in range(1,num+1):
            if sum([int(j) for j in str(i)]) % 2 == 0:
                ct += 1
        return ct
            
# 766. Toeplitz Matrix
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        # find the index
        for i in range(len(matrix)-1):
            for j in range(len(matrix[i])-1):
                if matrix[i][j] != matrix[i+1][j+1]:
                    return False
        return True
        
# 2259. Remove Digit From Number to Maximize Result
class Solution:
    def removeDigit(self, number: str, digit: str) -> str:
        idx = [i for i in range(len(number)) if number[i] == digit]
        if len(idx) == 0: return number
        ans = []
        for i in range(len(idx)):
            temp = list(number)
            temp.pop(idx[i])
            ans.append(int("".join(temp)))
        return str(max(ans))

# 1491. Average Salary Excluding the Minimum and Maximum Salary
class Solution:
    def average(self, salary: List[int]) -> float:
        ct = len(salary) - 2
        salary.sort()
        return sum(salary[1:-1]) / ct

# 191. Number of 1 Bits
class Solution:
    def hammingWeight(self, n: int) -> int:
        temp = 0
        while (n!=0):
            # in here, we flip the 1s to 0s and count the times we flip
            temp += 1
            # n&=5 --> n = n & 5
            n &= (n-1)
            print([n])
        return temp

# 976. Largest Perimeter Triangle
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        # revese the index
        for i in range(len(nums)-3,-1,-1):
            print(i)
            if nums[i] + nums[i+1] > nums[i+2]:
                return nums[i] + nums[i+1] + nums[i+2]
        return 0
#         ans = []
#         while len(nums) >= 3:
#             # print(nums[1] + nums[2] < nums[0])
#             temp = nums[-3:]
#             if (temp[1] + temp[2] > temp[0] and  temp[2] - temp[1] < temp[0]) == False:
#                 ans.append(0)
#                 nums = nums[:-3]
#             else:
#                 ans.append(sum(temp))
#                 nums = nums[:-3]
#         return max(ans)
        
# 1779. Find Nearest Point That Has the Same X or Y Coordinate
class Solution:
    def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
        ans = []
        dist = []
        for i in range(len(points)):
            if points[i][0] == x or points[i][1] == y:
                ans.append(i)
                d = abs(points[i][0] - x) + abs(points[i][1] - y)
                dist.append(d)
        if len(ans) == 0: return -1
        return ans[dist.index(min(dist))]

# 1502. Can Make Arithmetic Progression From Sequence
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        arr.sort()
        # def diff(a,b):
        #     return b-a
        sol = []
        for i in range(len(arr)-1,0,-1):
            sol.append(arr[i] - arr[i-1])
        return len(set(sol)) == 1

# 1790. Check if One String Swap Can Make Strings Equal
class Solution:
    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        # if len(s1) != len(s2): return False
        temp = {}
        ct = 0
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                ct += 1
                if s1[i] not in temp:
                    temp[s1[i]] = s2[i]
                # else:
                #     temp[s1[i]] += s2[i]
                
        print(temp)
        if ct > 2: return False
  
        return list(temp.keys())[::-1] == list(temp.values())

# 1299. Replace Elements with Greatest Element on Right Side
class Solution:
    def replaceElements(self, arr: List[int]) -> List[int]:
        ### The code below has time exceeds limit problem ###
        # for i in range(len(arr)-1):
        #     temp = arr[i]
        #     if max(arr[i:]) >= temp:
        #         arr[i] = max(arr[i+1:])
        # arr[-1] = -1
        # return arr
        max_val = -1
        for i in range(len(arr)-1, -1,-1):
            max_val,arr[i] = max(max_val,arr[i]), max_val
        return arr
# 605. Can Place Flowers
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        ct = 0
        # add a dummy head and dummy tail
        flowerbed = [0] + flowerbed
        flowerbed = flowerbed + [0]
        for i in range(1,len(flowerbed)-1):
            if flowerbed[i-1] == 0 and flowerbed[i+1] == 0:
                if flowerbed[i] == 0:
                    ct += 1
                    flowerbed[i] = 1
        return ct >= n

# 169. Majority Element
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        hash_table = {}
        for i in nums:
            if i not in hash_table:
                hash_table[i] = 1
            else:
                hash_table[i] += 1
        return max(hash_table, key = hash_table.get)

# 448. Find All Numbers Disappeared in an Array
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        full_lst = [i for i in range(1,len(nums) + 1)]

#         return [i for i in full_lst if i not in nums]
        nums.sort()
        return list(set(full_lst).difference(set(nums)))


# 1189. Maximum Number of Balloons
class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        hash_set = {'b':0,'a':0,'l':0,'o':0,'n':0}
        for i in text:
            if i in hash_set:
                hash_set[i] += 1
        double = min(hash_set['l'], hash_set['o'])
        single =  min(min(hash_set['b'], hash_set['a'], hash_set['n']),double // 2)
        print(hash_set)
        return single
        # find the largest common number in l and o key
        # and check if b a and n : has the number / 2
        # and check if b a and n : has the number / 2ß
# 2367. Number of Arithmetic Triplets
class Solution:
    def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
        # l,r = 0,1
        n = len(nums)
        ct = 0
        for i in range(n):
            if nums[i] + diff in nums and nums[i] + 2*diff in nums:
                ct += 1
        return ct
# 821. Shortest Distance to a Character
class Solution:
    def shortestToChar(self, s: str, c: str) -> List[int]:
        idx = []
        res = []
        for i in range(len(s)):
            if s[i] == c:
                idx.append(i)
        for i in range(len(s)):
            temp = 100000
            for j in idx:
                if j - i < temp:
                    temp  = abs(j-i)
            res.append(temp)
        return res
            
  # Solution online that is smart
class Solution:
    def shortestToChar(self, s: str, c: str) -> List[int]:
#         Travelling front to back
        result = ["*"] * len(s)
        i, j = 0, 0
        while i < len(s) and j < len(s):
            if s[i] == s[j] == c:
                result[i] = 0
                i += 1
                j += 1
            elif s[i] != c and s[j] == c:
                result[i] = abs(i-j)
                i += 1
            elif s[i] != c and s[j] != c:
                j += 1
    
#         Travelling back to front
        i = j = len(s) - 1
        while i >= 0 and j >= 0:
            if s[i] == s[j] == c:
                result[i] = 0
                i -= 1
                j -= 1
            elif s[i] != c and s[j] == c:
                if type(result[i]) == int:
                    result[i] = min(result[i], abs(i-j))
                else:
                    result[i] = abs(i-j)
                i -= 1
            elif s[i] != c and s[j] != c:
                j -= 1
        
        return result      


# 796. Rotate String
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        temp = ["" for i in range(len(s))]
        for i in range(0,len(s)):
            temp[i] = s[i:] + s[:i]
        #temp.append(s[len(s)-1:] + s[:len(s)-1])
        print(temp)
        return goal in temp

# 139. Word Break
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # temp = 0
        # while temp < len(s):
        #     for i in wordDict:
        #         if i == s[:len(i)]:
        #             temp += len(i)
        #             s = s[len(i):]
        # print(temp)
        # return temp
        # return s[temp:] in wordDict
        # while len(s) > 0 :
        #     print(True)
        #     for i in wordDict:
        #         if i == s[:len(i)]:
        #             s = s[len(i):]
        # print([s,'runned'])
        # return True if len(s) == 1 else False
        
        # to make sure this is a set, no duplicate words
        word = set(wordDict)
        # make a default dp list and everything is False unless we later on change it to True --> get it to the next index with a new word to check
        dp = [False] * (len(s) + 1)
        # we should start from index 0
        dp[0] = True
        # loop through every possible length of the string s
        for i in range(1,len(s) + 1):
            for j in range(i):
                # it comtains the combination: 0 1; ,,,; 1 2; 0 2; 0 8; 3 8; ... etc
                if dp[j] and s[j:i] in word:
                    dp[i] = True
                    break
        print(dp)       
        return dp[len(s)] # why to check the last index


# 746. Min Cost Climbing Stairs
class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        dp = [0] * (len(cost) + 1)
        for i in range(2,len(cost)+1):
            dp[i] = min(cost[i-1] + dp[i-1],cost[i-2] + dp[i-2])
        return dp[-1]

# 1137. N-th Tribonacci Number
class Solution(object):
    def tribonacci(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [0] *  (n + 1)
        if n > 2:
            dp[0] = 0
            dp[1] = 1
            dp[2] = 1
        else: return [0,1,1][n]
        for i in range(3,n+1):
            #print([i])
            dp[i] = dp[i-1] + dp[i-2] + dp[i-3]
        return dp[n]

# 1646. Get Maximum in Generated Array
class Solution:
    def getMaximumGenerated(self, n: int) -> int:
        dp = [0] * (n+1)
        if n >= 2:
            dp[0] = 0
            dp[1] = 1
            if n % 2 == 0:
                for i in range(1,n//2):
                    dp[2 * i] = dp[i]
                    dp[2 * i + 1] = dp[i] + dp[i + 1]
            else:
                 for i in range(1,n//2+1):
                    dp[2 * i] = dp[i]
                    dp[2 * i + 1] = dp[i] + dp[i + 1]
        else: return [0,1][n]
        return max(dp)
    # the case of 2 does not work, ok works now
# 198. House Robber
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums: return 0
        N = len(nums)
        dp = [0] * (len(nums) + 1)
        
        dp[N-1] = nums[N-1]
        
        for i in range(N-2,-1,-1):
            dp[i] = max(dp[i+1],dp[i+2]+nums[i])
            
        return dp[0]
# 1544. Make The String Great
class Solution:
    def makeGood(self, s: str) -> str:  
        while len(s) > 1:
            find = False
            
            for i in range(len(s)-1):
                curr,next_ = s[i],s[i+1]
                
                if abs(ord(curr) - ord(next_)) == 32:
                    s = s[:i] + s[i+2:]
                    find = True
                    break
            if not find:
                break
        return s
# 2418. Sort the People
class Solution(object):
    def sortPeople(self, names, heights):
        """
        :type names: List[str]
        :type heights: List[int]
        :rtype: List[str]
        """
        temp = sorted(heights)[::-1]
        idx = []
        for i in temp:
            idx.append(heights.index(i))
        return [names[j] for j in idx]
# 1816. Truncate Sentence
class Solution(object):
    def truncateSentence(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        return ' '.join(s.split(' ')[:k])
# 1684. Count the Number of Consistent Strings
class Solution(object):
    def countConsistentStrings(self, allowed, words):
        """
        :type allowed: str
        :type words: List[str]
        :rtype: int
        """
        nums = 0
        for i in words:
            ct = 0
            for j in i:
                if j in allowed:
                    ct += 1
            if ct == len(i):
                nums += 1
        return nums
# 1534. Count Good Triplets
class Solution(object):
    def countGoodTriplets(self, arr, a, b, c):
        """
        :type arr: List[int]
        :type a: int
        :type b: int
        :type c: int
        :rtype: int
        """
        ct = 0
        for i in range(len(arr)):
            for j in range(i+1,len(arr)):
                for k in range(j+1,len(arr)):
                    if (abs(arr[i] - arr[j]) <= a) & (abs(arr[j] - arr[k]) <= b) & (abs(arr[i] - arr[k]) <= c):
                        ct += 1
        return ct

## 1133. Largest Unique Number
class Solution:
    def largestUniqueNumber(self, nums: List[int]) -> int:
        nums.sort()
        nums = nums[::-1]
        temp = {}
        for i in nums:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] += 1
                
        for k,v in temp.items():
            if v == 1:
                return k
        return -1
# 1991. Find the Middle Index in Array
class Solution:
    def findMiddleIndex(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            l = nums[:i]
            r = nums[i+1:]
            if sum(l) == sum(r):
                return i
        return -1         
# 1672. Richest Customer Wealth
class Solution(object):
    def maximumWealth(self, accounts):
        """
        :type accounts: List[List[int]]
        :rtype: int
        """
        return max([sum(i) for i in accounts])
# 1342. Number of Steps to Reduce a Number to Zero
class Solution(object):
    def numberOfSteps(self, num):
        """
        :type num: int
        :rtype: int
        """
        ct = 0
        while num!=0:
            if num % 2 == 0:
                num /=2
            else:
                num -=1
            ct +=1
        return ct

# 876. Middle of the Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        temp = head
        ct = 1
        while temp.next!= None:
            temp = temp.next
            ct += 1
        for i in range(ct // 2):
            head = head.next
        return head
#         output = head
#         if ct % 2 == 0:
#             for i in range(ct // 2):
#                 head = head.next
#             return head
#         else:
#             for i in range((ct) // 2):
#                 output = output.next
#             return output
            
        
# 1196. How Many Apples Can You Put into the Basket
class Solution:
    def maxNumberOfApples(self, weight: List[int]) -> int:
        max_weight = 5000
        ct = 0
        weight.sort()
        # summ = weight[0]
        # while summ < max_weight:
        #     ct += 1
        #     weight = weight[1:]
        #     summ += weight[0]
        # return ct
        for i in range(len(weight)):
            if sum(weight[:i+1]) <= max_weight:
                ct +=1
        return ct


# 2164. Sort Even and Odd Indices Independently
class Solution:
    def sortEvenOdd(self, nums: List[int]) -> List[int]:
        output = []
        lst1 = sorted([nums[i] for i in range(0,len(nums),2)])
        lst2 = sorted([nums[i] for i in range(1,len(nums),2)])[::-1]
        for i in range(min(len(lst1),len(lst2))):
            output.append(lst1[i])
            output.append(lst2[i])
        if len(lst1) > len(lst2):
            output.append(lst1[-1])
        if len(lst2) > len(lst1):
            output.append(lst2[-1])
        return output

# 2073. Time Needed to Buy Tickets
class Solution:
    def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
        ct = 0
        while True:
            for i in range(len(tickets)):
                if tickets[i] > 0:
                    ct += 1
                    tickets[i] -=1
                else:
                    continue
                if tickets[k] == 0:
                    return ct
# 1385. Find the Distance Value Between Two Arrays
class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        # if len(arr2) < len(arr1):
        #     arr2,arr1 = arr1,arr2
        l,r = 0, len(arr2)
        output = []
        for i in range(len(arr1)):
            temp = []
            while l < r :
                temp.append(abs(arr1[i] - arr2[l]) > d)
                l +=1
            #output.append(temp)
            output.append(all([i for i in temp]))
            l = 0
        print(output)
        return len([i for i in output if i == True])