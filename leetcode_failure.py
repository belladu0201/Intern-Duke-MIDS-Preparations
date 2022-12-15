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
# 409. Longest Palindrome
class Solution:
    def longestPalindrome(self, s: str) -> int:
        # if len(s) % 2 == 0:
        #     # 4, index 1,2
        #     l1,l2 = len(s)//2 - 1,len(s)//2
        #     max_len = 0
        # else:
        #     # 3, index 0,2
        #     l1,l2 = len(s)//2 - 1,len(s)//2 + 1
        #     max_len = 1
        # while l1 > 0 and l2 < len(s):
        #     if s[l1] == s[l2]:
        #         max_len += 2
        #         l1 -= 1
        #         l2 += 1
        #     else:
        #         break
        # return max_len
        temp = {}
        result = 0
        for i in s:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] += 1
        for k,v in temp.items():
            if v % 2 == 0:
                result += v
            else:
                while v - 1 > 0:
                    result += 2
        if result == len(s):
            return result
        # if len(s) % 2 == 0:
        #     return result + 1
        
        return result + 1
# 1710. Maximum Units on a Truck
class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
#         temp = {}
#         for i in boxTypes:
#             if i[1] not in temp:
#                 temp[i[1]] = i[0]
#             else:
#                 temp[i[1]] += i[0]
#         temp = sorted(temp.items())

#         return temp
    
        ct = 0
        output = 0
        stop = []
        boxTypes.sort(key= lambda x: x[1], reverse=True) #sorting the list by unit count
        for i in boxTypes:
            #print([i[0]])
            if ct + i[0] <= truckSize:
                output += i[0] * i[1]
                #print([i[0]])
                ct += i[0]
                #print([output])
                stop.append(boxTypes.index(i))
            else: break
        #print([ct,output])
        print(stop[-1])
        if ct < truckSize:
        #if (stop[-1] + 1) < len(boxTypes):
            print("---------")
            #print([ct,truckSize,output,stop])
            #print(boxTypes[boxTypes[stop[-1]] + 1])
            #output += (truckSize - ct) * (boxTypes[boxTypes[stop[-1] + 1]][1])
        return output

# 976. Largest Perimeter Triangle
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        ans = []
        while len(nums) >= 3:
            # print(nums[1] + nums[2] < nums[0])
            temp = nums[-3:]
            if (temp[1] + temp[2] > temp[0] and  temp[2] - temp[1] < temp[0]) == False:
                ans.append(0)
                nums = nums[:-3]
            else:
                ans.append(sum(temp))
                nums = nums[:-3]
        return max(ans)
# 2446. Determine if Two Events Have Conflict
class Solution(object):
    def haveConflict(self, event1, event2):
        """
        :type event1: List[str]
        :type event2: List[str]
        :rtype: bool
        """
        print(int(event1[1][:2]))
        print(int(event2[0][:2]))
        if int(event2[0][:2]) < int(event1[1][:2]):
            print("Run1")
            return True if (int(event2[0][:2]) and int(event1[1][:2]) <= 12) else False
        elif int(event2[0][:2]) == int(event1[1][:2]):
            if int(event1[1][-2:]) <= int(event2[0][-2:]):
                print("Run2")
                return True
            print("Run3")
            return False
        else:
            print("Run4")
            return False