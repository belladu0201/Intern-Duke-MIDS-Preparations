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
        
