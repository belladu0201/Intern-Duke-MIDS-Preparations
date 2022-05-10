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
