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
