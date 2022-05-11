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
        
# return n-1
