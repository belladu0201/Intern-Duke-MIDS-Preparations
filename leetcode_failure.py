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
