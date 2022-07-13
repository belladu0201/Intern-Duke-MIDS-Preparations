# Medium: longest-substring-without-repeating-characters
# F A I L U R E CODE!!!
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) == 1:
            return 1
        if len(s) == 0:
            return 0
        temp  = {}
        length = []
        switch = 0
        for i in s:
            #print(temp)
            if i not in temp:
                #print(i)
                temp[i] = 1
                #print(temp)
            else:
                print(temp,"last")
                switch += 1
                length.append(len(temp.values()))
                temp  = {}
                temp[i] = 1
                #print([temp,'new'])
            length.append(len(temp.values()))
        if switch >= 1:
            return max(length)
        return len(temp.values())
##################################################
        temp = {}
        idx = [0]
        for i in range(len(s)):
            if s[i] not in temp:
                temp[s[i]] = 1
            else:
                idx.append(i)
                temp = {}
                temp[s[i]] = 1
        #idx.append(len(s))
        output = [i for i in range(len(idx))]
        for i in range(len(idx)-1):
            output.append(idx[i:i+1])
        print(output)

        #return (max(diff))
################################################## Right Answer using recursion
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        def find_re(s):
            if len(s) == 1:
                return 1
            if len(s) == 0:
                return 0
            temp = {}
            repeated = []
            count = 0
            start_idx = 0
            for i, elem in enumerate(s):
                start_idx = i
                if elem not in temp:
                    temp[elem] = 1
                else:
                    count += 1
                    repeated.append(elem) # key that repeated
                    break
            if count == 0:
                return len(s)
            return max(start_idx,find_re(s[1:]))
        return find_re(s)

    # 152. Maximum Product Subarray
    class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        counter = 0
        lst = []
        prod = 1
        if len(nums) == 1:
            return nums[0]
        for i in nums:
            if i<0:
                counter += 1
                lst.append(i)
        if counter == 0:
            for i in nums:
                if i != 0:
                    prod *= i
        elif counter % 2 == 0:
            for i in nums:
                prod*=i
        else:
            #prod1,prod2,prod3,prod4 = 1,1,1,1
            temp1 =nums[:nums.index(lst[0])+1]
            temp2 =nums[nums.index(lst[0])+1:]
            temp3 = nums[:nums.index(lst[-1])+1]
            temp4 = nums[nums.index(lst[-1])+1:]
            print([temp1, temp2,temp3,temp4])
            for i in temp1:
                prod *= i
            #print(prod)
            lst.append(prod)
            prod = 1
            for j in temp2:
                prod *= j
            lst.append(prod)
            prod = 1
            for k in temp3:
                prod *= k
            lst.append(prod)
            prod = 1
            for l in temp4:
                prod *= l
            lst.append(prod)
            print(lst)
            prod = max(lst)
        return prod
            
   # 680
class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        left,right = 0,len(s)-1
        count_delete = 0
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -=1
            elif count_delete == 0:
                if s[left+1] == s[right] :
                    count_delete+=1
                    left +=2
                if s[left] == s[right-1]:
                    count_delete+=1
                    right -=2
            elif s[left] != s[right]:
                return False
        return True
