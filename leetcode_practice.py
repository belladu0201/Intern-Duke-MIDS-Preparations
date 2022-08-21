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
