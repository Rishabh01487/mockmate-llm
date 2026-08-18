"""
gen_sample_dataset.py
====================

Generates a small synthetic leetcode_dataset.jsonl (12 problems × 4 languages
= 48 rows) so we can smoke-test prepare_data.py and the downstream API/UI
without needing the real 5,000-problem corpus.

Run:
    python gen_sample_dataset.py --out ./leetcode_dataset.jsonl --rows 48
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# --- 12 canonical LeetCode problems (abbreviated) --------------------------
PROBLEMS = [
    {
        "problem_id": 1, "title": "Two Sum", "difficulty": "Easy",
        "description": "Given an array of integers nums and an integer target, "
                       "return indices of the two numbers such that they add up to target. "
                       "You may assume each input has exactly one solution and you may not "
                       "use the same element twice.",
        "examples": "Input: nums = [2,7,11,15], target = 9\nOutput: [0,1]\nExplanation: nums[0] + nums[1] == 9.",
        "constraints": "2 <= nums.length <= 10^4\n-10^9 <= nums[i] <= 10^9\nOnly one valid answer exists.",
    },
    {
        "problem_id": 2, "title": "Add Two Numbers", "difficulty": "Medium",
        "description": "You are given two non-empty linked lists representing two non-negative integers. "
                       "The digits are stored in reverse order and each of their nodes contains a single digit. "
                       "Add the two numbers and return the sum as a linked list.",
        "examples": "Input: l1 = [2,4,3], l2 = [5,6,4]\nOutput: [7,0,8]\nExplanation: 342 + 465 = 807.",
        "constraints": "The number of nodes in each list is in [1, 100].\n0 <= Node.val <= 9.",
    },
    {
        "problem_id": 3, "title": "Longest Substring Without Repeating Characters", "difficulty": "Medium",
        "description": "Given a string s, find the length of the longest substring without repeating characters.",
        "examples": "Input: s = 'abcabcbb'\nOutput: 3\nExplanation: The answer is 'abc', with length 3.",
        "constraints": "0 <= s.length <= 5 * 10^4.",
    },
    {
        "problem_id": 5, "title": "Longest Palindromic Substring", "difficulty": "Medium",
        "description": "Given a string s, return the longest palindromic substring in s.",
        "examples": "Input: s = 'babad'\nOutput: 'bab'.\nExplanation: 'aba' is also a valid answer.",
        "constraints": "1 <= s.length <= 1000.",
    },
    {
        "problem_id": 7, "title": "Reverse Integer", "difficulty": "Medium",
        "description": "Given a signed 32-bit integer x, return x with its digits reversed. "
                       "If reversing causes overflow, return 0.",
        "examples": "Input: x = 123\nOutput: 321\nInput: x = -123\nOutput: -321.",
        "constraints": "-2^31 <= x <= 2^31 - 1",
    },
    {
        "problem_id": 9, "title": "Palindrome Number", "difficulty": "Easy",
        "description": "Given an integer x, return true if x is a palindrome, false otherwise.",
        "examples": "Input: x = 121\nOutput: true\nInput: x = -121\nOutput: false.",
        "constraints": "-2^31 <= x <= 2^31 - 1",
    },
    {
        "problem_id": 11, "title": "Container With Most Water", "difficulty": "Medium",
        "description": "You are given an integer array height of length n. Find two lines that "
                       "together with the x-axis form a container holding the most water. Return the maximum amount of water.",
        "examples": "Input: height = [1,8,6,2,5,4,8,3,7]\nOutput: 49.",
        "constraints": "n >= 2\n0 <= height[i] <= 10^4.",
    },
    {
        "problem_id": 13, "title": "Roman to Integer", "difficulty": "Easy",
        "description": "Given a roman numeral, convert it to an integer.",
        "examples": "Input: s = 'III'\nOutput: 3\nInput: s = 'LVIII'\nOutput: 58.",
        "constraints": "1 <= s.length <= 15.",
    },
    {
        "problem_id": 14, "title": "Longest Common Prefix", "difficulty": "Easy",
        "description": "Write a function to find the longest common prefix string amongst an array of strings. "
                       "If there is no common prefix, return an empty string.",
        "examples": "Input: strs = ['flower','flow','flight']\nOutput: 'fl'.",
        "constraints": "1 <= strs.length <= 200.",
    },
    {
        "problem_id": 15, "title": "3Sum", "difficulty": "Medium",
        "description": "Given an integer array nums, return all triplets [nums[i], nums[j], nums[k]] "
                       "such that i != j, j != k, i != k and nums[i]+nums[j]+nums[k] == 0.",
        "examples": "Input: nums = [-1,0,1,2,-1,-4]\nOutput: [[-1,-1,2],[-1,0,1]].",
        "constraints": "3 <= nums.length <= 3000.",
    },
    {
        "problem_id": 20, "title": "Valid Parentheses", "difficulty": "Easy",
        "description": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', "
                       "determine if the input string is valid.",
        "examples": "Input: s = '()'\nOutput: true\nInput: s = '([)]'\nOutput: false.",
        "constraints": "1 <= s.length <= 10^4.",
    },
    {
        "problem_id": 53, "title": "Maximum Subarray", "difficulty": "Medium",
        "description": "Given an integer array nums, find the contiguous subarray with the largest sum "
                       "and return its sum.",
        "examples": "Input: nums = [-2,1,-3,4,-1,2,1,-5,4]\nOutput: 6.",
        "constraints": "1 <= nums.length <= 10^5.",
    },
]

# --- 4 language solutions per problem ------------------------------------
# Note: these are intentionally simple/correct reference solutions so the
# heuristic complexity analyzer in prepare_data.py produces sensible Big-O.

PY_SOLUTIONS = {
    1:  "class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n        for i, n in enumerate(nums):\n            if target - n in seen:\n                return [seen[target - n], i]\n            seen[n] = i",
    2:  "class Solution:\n    def addTwoNumbers(self, l1, l2):\n        carry = 0\n        dummy = ListNode(0)\n        cur = dummy\n        while l1 or l2 or carry:\n            v = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry\n            cur.next = ListNode(v % 10)\n            carry = v // 10\n            cur = cur.next\n            if l1: l1 = l1.next\n            if l2: l2 = l2.next\n        return dummy.next",
    3:  "class Solution:\n    def lengthOfLongestSubstring(self, s):\n        seen = {}\n        l = 0\n        best = 0\n        for r, c in enumerate(s):\n            if c in seen and seen[c] >= l:\n                l = seen[c] + 1\n            seen[c] = r\n            best = max(best, r - l + 1)\n        return best",
    5:  "class Solution:\n    def longestPalindrome(self, s):\n        def expand(a, b):\n            while a >= 0 and b < len(s) and s[a] == s[b]:\n                a -= 1; b += 1\n            return s[a+1:b]\n        best = ''\n        for i in range(len(s)):\n            odd = expand(i, i)\n            even = expand(i, i+1)\n            if len(odd) > len(best): best = odd\n            if len(even) > len(best): best = even\n        return best",
    7:  "class Solution:\n    def reverse(self, x):\n        sign = -1 if x < 0 else 1\n        x = abs(x)\n        rev = 0\n        while x:\n            rev = rev * 10 + x % 10\n            x //= 10\n        rev *= sign\n        return rev if -2**31 <= rev <= 2**31 - 1 else 0",
    9:  "class Solution:\n    def isPalindrome(self, x):\n        if x < 0: return False\n        s = str(x)\n        return s == s[::-1]",
    11: "class Solution:\n    def maxArea(self, height):\n        l, r = 0, len(height) - 1\n        best = 0\n        while l < r:\n            best = max(best, (r - l) * min(height[l], height[r]))\n            if height[l] < height[r]: l += 1\n            else: r -= 1\n        return best",
    13: "class Solution:\n    def romanToInt(self, s):\n        m = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n        total = 0\n        for i in range(len(s)):\n            if i+1 < len(s) and m[s[i]] < m[s[i+1]]:\n                total -= m[s[i]]\n            else:\n                total += m[s[i]]\n        return total",
    14: "class Solution:\n    def longestCommonPrefix(self, strs):\n        if not strs: return ''\n        pref = strs[0]\n        for w in strs[1:]:\n            while not w.startswith(pref):\n                pref = pref[:-1]\n                if not pref: return ''\n        return pref",
    15: "class Solution:\n    def threeSum(self, nums):\n        nums.sort()\n        out = []\n        for i in range(len(nums) - 2):\n            if i > 0 and nums[i] == nums[i-1]: continue\n            l, r = i+1, len(nums)-1\n            while l < r:\n                s = nums[i] + nums[l] + nums[r]\n                if s < 0: l += 1\n                elif s > 0: r -= 1\n                else:\n                    out.append([nums[i], nums[l], nums[r]])\n                    while l < r and nums[l] == nums[l+1]: l += 1\n                    while l < r and nums[r] == nums[r-1]: r -= 1\n                    l += 1; r -= 1\n        return out",
    20: "class Solution:\n    def isValid(self, s):\n        pair = {')': '(', ']': '[', '}': '{'}\n        stack = []\n        for c in s:\n            if c in pair:\n                if not stack or stack.pop() != pair[c]: return False\n            else: stack.append(c)\n        return not stack",
    53: "class Solution:\n    def maxSubArray(self, nums):\n        best = cur = nums[0]\n        for n in nums[1:]:\n            cur = max(n, cur + n)\n            best = max(best, cur)\n        return best",
}

JAVA_SOLUTIONS = {
    1:  "class Solution {\n    public int[] twoSum(int[] nums, int target) {\n        Map<Integer,Integer> map = new HashMap<>();\n        for (int i = 0; i < nums.length; i++) {\n            int comp = target - nums[i];\n            if (map.containsKey(comp)) return new int[]{map.get(comp), i};\n            map.put(nums[i], i);\n        }\n        return new int[0];\n    }\n}",
    2:  "class Solution {\n    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {\n        ListNode dummy = new ListNode(0), cur = dummy;\n        int carry = 0;\n        while (l1 != null || l2 != null || carry != 0) {\n            int v = (l1 != null ? l1.val : 0) + (l2 != null ? l2.val : 0) + carry;\n            cur.next = new ListNode(v % 10);\n            carry = v / 10;\n            cur = cur.next;\n            if (l1 != null) l1 = l1.next;\n            if (l2 != null) l2 = l2.next;\n        }\n        return dummy.next;\n    }\n}",
    3:  "class Solution {\n    public int lengthOfLongestSubstring(String s) {\n        Map<Character,Integer> seen = new HashMap<>();\n        int l = 0, best = 0;\n        for (int r = 0; r < s.length(); r++) {\n            char c = s.charAt(r);\n            if (seen.containsKey(c) && seen.get(c) >= l) l = seen.get(c) + 1;\n            seen.put(c, r);\n            best = Math.max(best, r - l + 1);\n        }\n        return best;\n    }\n}",
    5:  "class Solution {\n    public String longestPalindrome(String s) {\n        String best = \"\";\n        for (int i = 0; i < s.length(); i++) {\n            String odd = expand(s, i, i), even = expand(s, i, i+1);\n            if (odd.length() > best.length()) best = odd;\n            if (even.length() > best.length()) best = even;\n        }\n        return best;\n    }\n    private String expand(String s, int a, int b) {\n        while (a >= 0 && b < s.length() && s.charAt(a) == s.charAt(b)) { a--; b++; }\n        return s.substring(a+1, b);\n    }\n}",
    7:  "class Solution {\n    public int reverse(int x) {\n        int rev = 0;\n        while (x != 0) {\n            int d = x % 10; x /= 10;\n            if (rev > Integer.MAX_VALUE/10 || (rev == Integer.MAX_VALUE/10 && d > 7)) return 0;\n            if (rev < Integer.MIN_VALUE/10 || (rev == Integer.MIN_VALUE/10 && d < -8)) return 0;\n            rev = rev * 10 + d;\n        }\n        return rev;\n    }\n}",
    9:  "class Solution {\n    public boolean isPalindrome(int x) {\n        if (x < 0) return false;\n        String s = Integer.toString(x);\n        int l = 0, r = s.length() - 1;\n        while (l < r) {\n            if (s.charAt(l) != s.charAt(r)) return false;\n            l++; r--;\n        }\n        return true;\n    }\n}",
    11: "class Solution {\n    public int maxArea(int[] height) {\n        int l = 0, r = height.length - 1, best = 0;\n        while (l < r) {\n            best = Math.max(best, (r - l) * Math.min(height[l], height[r]));\n            if (height[l] < height[r]) l++; else r--;\n        }\n        return best;\n    }\n}",
    13: "class Solution {\n    public int romanToInt(String s) {\n        Map<Character,Integer> m = Map.of('I',1,'V',5,'X',10,'L',50,'C',100,'D',500,'M',1000);\n        int total = 0;\n        for (int i = 0; i < s.length(); i++) {\n            int v = m.get(s.charAt(i));\n            if (i+1 < s.length() && v < m.get(s.charAt(i+1))) total -= v;\n            else total += v;\n        }\n        return total;\n    }\n}",
    14: "class Solution {\n    public String longestCommonPrefix(String[] strs) {\n        if (strs.length == 0) return \"\";\n        String pref = strs[0];\n        for (int i = 1; i < strs.length; i++) {\n            while (!strs[i].startsWith(pref)) {\n                pref = pref.substring(0, pref.length() - 1);\n                if (pref.isEmpty()) return \"\";\n            }\n        }\n        return pref;\n    }\n}",
    15: "class Solution {\n    public List<List<Integer>> threeSum(int[] nums) {\n        Arrays.sort(nums);\n        List<List<Integer>> out = new ArrayList<>();\n        for (int i = 0; i < nums.length - 2; i++) {\n            if (i > 0 && nums[i] == nums[i-1]) continue;\n            int l = i+1, r = nums.length-1;\n            while (l < r) {\n                int s = nums[i] + nums[l] + nums[r];\n                if (s < 0) l++; else if (s > 0) r--;\n                else {\n                    out.add(List.of(nums[i], nums[l], nums[r]));\n                    while (l < r && nums[l] == nums[l+1]) l++;\n                    while (l < r && nums[r] == nums[r-1]) r--;\n                    l++; r--;\n                }\n            }\n        }\n        return out;\n    }\n}",
    20: "class Solution {\n    public boolean isValid(String s) {\n        Map<Character,Character> pair = Map.of(')', '(', ']', '[', '}', '{');\n        Deque<Character> stack = new ArrayDeque<>();\n        for (char c : s.toCharArray()) {\n            if (pair.containsKey(c)) {\n                if (stack.isEmpty() || stack.pop() != pair.get(c)) return false;\n            } else stack.push(c);\n        }\n        return stack.isEmpty();\n    }\n}",
    53: "class Solution {\n    public int maxSubArray(int[] nums) {\n        int best = nums[0], cur = nums[0];\n        for (int i = 1; i < nums.length; i++) {\n            cur = Math.max(nums[i], cur + nums[i]);\n            best = Math.max(best, cur);\n        }\n        return best;\n    }\n}",
}

CPP_SOLUTIONS = {
    1:  "class Solution {\npublic:\n    vector<int> twoSum(vector<int>& nums, int target) {\n        unordered_map<int,int> seen;\n        for (int i = 0; i < nums.size(); ++i) {\n            int comp = target - nums[i];\n            if (seen.count(comp)) return {seen[comp], i};\n            seen[nums[i]] = i;\n        }\n        return {};\n    }\n};",
    2:  "class Solution {\npublic:\n    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {\n        ListNode dummy(0); ListNode* cur = &dummy; int carry = 0;\n        while (l1 || l2 || carry) {\n            int v = (l1 ? l1->val : 0) + (l2 ? l2->val : 0) + carry;\n            cur->next = new ListNode(v % 10); carry = v / 10;\n            cur = cur->next;\n            if (l1) l1 = l1->next;\n            if (l2) l2 = l2->next;\n        }\n        return dummy.next;\n    }\n};",
    3:  "class Solution {\npublic:\n    int lengthOfLongestSubstring(string s) {\n        unordered_map<char,int> seen; int l = 0, best = 0;\n        for (int r = 0; r < s.size(); ++r) {\n            char c = s[r];\n            if (seen.count(c) && seen[c] >= l) l = seen[c] + 1;\n            seen[c] = r;\n            best = max(best, r - l + 1);\n        }\n        return best;\n    }\n};",
    5:  "class Solution {\npublic:\n    string longestPalindrome(string s) {\n        string best;\n        for (int i = 0; i < s.size(); ++i) {\n            string odd = expand(s, i, i), even = expand(s, i, i+1);\n            if (odd.size() > best.size()) best = odd;\n            if (even.size() > best.size()) best = even;\n        }\n        return best;\n    }\nprivate:\n    string expand(const string& s, int a, int b) {\n        while (a >= 0 && b < s.size() && s[a] == s[b]) { --a; ++b; }\n        return s.substr(a+1, b - a - 1);\n    }\n};",
    7:  "class Solution {\npublic:\n    int reverse(int x) {\n        long rev = 0;\n        while (x) { rev = rev * 10 + x % 10; x /= 10; }\n        return (rev > INT_MAX || rev < INT_MIN) ? 0 : rev;\n    }\n};",
    9:  "class Solution {\npublic:\n    bool isPalindrome(int x) {\n        if (x < 0) return false;\n        string s = to_string(x);\n        int l = 0, r = s.size() - 1;\n        while (l < r) { if (s[l++] != s[r--]) return false; }\n        return true;\n    }\n};",
    11: "class Solution {\npublic:\n    int maxArea(vector<int>& height) {\n        int l = 0, r = height.size() - 1, best = 0;\n        while (l < r) {\n            best = max(best, (r - l) * min(height[l], height[r]));\n            if (height[l] < height[r]) l++; else r--;\n        }\n        return best;\n    }\n};",
    13: "class Solution {\npublic:\n    int romanToInt(string s) {\n        unordered_map<char,int> m = {{'I',1},{'V',5},{'X',10},{'L',50},{'C',100},{'D',500},{'M',1000}};\n        int total = 0;\n        for (int i = 0; i < s.size(); ++i) {\n            int v = m[s[i]];\n            if (i+1 < s.size() && v < m[s[i+1]]) total -= v;\n            else total += v;\n        }\n        return total;\n    }\n};",
    14: "class Solution {\npublic:\n    string longestCommonPrefix(vector<string>& strs) {\n        if (strs.empty()) return \"\";\n        string pref = strs[0];\n        for (int i = 1; i < strs.size(); ++i) {\n            while (strs[i].find(pref) != 0) {\n                pref.pop_back();\n                if (pref.empty()) return \"\";\n            }\n        }\n        return pref;\n    }\n};",
    15: "class Solution {\npublic:\n    vector<vector<int>> threeSum(vector<int>& nums) {\n        sort(nums.begin(), nums.end());\n        vector<vector<int>> out;\n        for (int i = 0; i + 2 < nums.size(); ++i) {\n            if (i > 0 && nums[i] == nums[i-1]) continue;\n            int l = i+1, r = nums.size()-1;\n            while (l < r) {\n                int s = nums[i] + nums[l] + nums[r];\n                if (s < 0) l++; else if (s > 0) r--;\n                else {\n                    out.push_back({nums[i], nums[l], nums[r]});\n                    while (l < r && nums[l] == nums[l+1]) l++;\n                    while (l < r && nums[r] == nums[r-1]) r--;\n                    l++; r--;\n                }\n            }\n        }\n        return out;\n    }\n};",
    20: "class Solution {\npublic:\n    bool isValid(string s) {\n        unordered_map<char,char> pair = {{')','('},{']','['},{'}','{'}};\n        vector<char> st;\n        for (char c : s) {\n            if (pair.count(c)) {\n                if (st.empty() || st.back() != pair[c]) return false;\n                st.pop_back();\n            } else st.push_back(c);\n        }\n        return st.empty();\n    }\n};",
    53: "class Solution {\npublic:\n    int maxSubArray(vector<int>& nums) {\n        int best = nums[0], cur = nums[0];\n        for (int i = 1; i < nums.size(); ++i) {\n            cur = max(nums[i], cur + nums[i]);\n            best = max(best, cur);\n        }\n        return best;\n    }\n};",
}

JS_SOLUTIONS = {
    1:  "/**\n * @param {number[]} nums\n * @param {number} target\n * @return {number[]}\n */\nvar twoSum = function(nums, target) {\n    const seen = new Map();\n    for (let i = 0; i < nums.length; i++) {\n        const comp = target - nums[i];\n        if (seen.has(comp)) return [seen.get(comp), i];\n        seen.set(nums[i], i);\n    }\n    return [];\n};",
    2:  "/**\n * @param {ListNode} l1\n * @param {ListNode} l2\n * @return {ListNode}\n */\nvar addTwoNumbers = function(l1, l2) {\n    let dummy = new ListNode(0), cur = dummy, carry = 0;\n    while (l1 || l2 || carry) {\n        const v = (l1 ? l1.val : 0) + (l2 ? l2.val : 0) + carry;\n        cur.next = new ListNode(v % 10); carry = Math.floor(v / 10);\n        cur = cur.next;\n        if (l1) l1 = l1.next;\n        if (l2) l2 = l2.next;\n    }\n    return dummy.next;\n};",
    3:  "/**\n * @param {string} s\n * @return {number}\n */\nvar lengthOfLongestSubstring = function(s) {\n    const seen = new Map();\n    let l = 0, best = 0;\n    for (let r = 0; r < s.length; r++) {\n        const c = s[r];\n        if (seen.has(c) && seen.get(c) >= l) l = seen.get(c) + 1;\n        seen.set(c, r);\n        best = Math.max(best, r - l + 1);\n    }\n    return best;\n};",
    5:  "/**\n * @param {string} s\n * @return {string}\n */\nvar longestPalindrome = function(s) {\n    let best = '';\n    const expand = (a, b) => {\n        while (a >= 0 && b < s.length && s[a] === s[b]) { a--; b++; }\n        return s.slice(a+1, b);\n    };\n    for (let i = 0; i < s.length; i++) {\n        const odd = expand(i, i), even = expand(i, i+1);\n        if (odd.length > best.length) best = odd;\n        if (even.length > best.length) best = even;\n    }\n    return best;\n};",
    7:  "/**\n * @param {number} x\n * @return {number}\n */\nvar reverse = function(x) {\n    const sign = x < 0 ? -1 : 1;\n    x = Math.abs(x);\n    let rev = 0;\n    while (x) { rev = rev * 10 + x % 10; x = Math.floor(x / 10); }\n    rev *= sign;\n    return (rev > 2**31 - 1 || rev < -(2**31)) ? 0 : rev;\n};",
    9:  "/**\n * @param {number} x\n * @return {boolean}\n */\nvar isPalindrome = function(x) {\n    if (x < 0) return false;\n    const s = x.toString();\n    let l = 0, r = s.length - 1;\n    while (l < r) { if (s[l] !== s[r]) return false; l++; r--; }\n    return true;\n};",
    11: "/**\n * @param {number[]} height\n * @return {number}\n */\nvar maxArea = function(height) {\n    let l = 0, r = height.length - 1, best = 0;\n    while (l < r) {\n        best = Math.max(best, (r - l) * Math.min(height[l], height[r]));\n        if (height[l] < height[r]) l++; else r--;\n    }\n    return best;\n};",
    13: "/**\n * @param {string} s\n * @return {number}\n */\nvar romanToInt = function(s) {\n    const m = {I:1,V:5,X:10,L:50,C:100,D:500,M:1000};\n    let total = 0;\n    for (let i = 0; i < s.length; i++) {\n        const v = m[s[i]];\n        if (i+1 < s.length && v < m[s[i+1]]) total -= v;\n        else total += v;\n    }\n    return total;\n};",
    14: "/**\n * @param {string[]} strs\n * @return {string}\n */\nvar longestCommonPrefix = function(strs) {\n    if (strs.length === 0) return '';\n    let pref = strs[0];\n    for (let i = 1; i < strs.length; i++) {\n        while (!strs[i].startsWith(pref)) {\n            pref = pref.slice(0, -1);\n            if (!pref) return '';\n        }\n    }\n    return pref;\n};",
    15: "/**\n * @param {number[]} nums\n * @return {number[][]}\n */\nvar threeSum = function(nums) {\n    nums.sort((a,b) => a - b);\n    const out = [];\n    for (let i = 0; i + 2 < nums.length; i++) {\n        if (i > 0 && nums[i] === nums[i-1]) continue;\n        let l = i+1, r = nums.length-1;\n        while (l < r) {\n            const s = nums[i] + nums[l] + nums[r];\n            if (s < 0) l++; else if (s > 0) r--; else {\n                out.push([nums[i], nums[l], nums[r]]);\n                while (l < r && nums[l] === nums[l+1]) l++;\n                while (l < r && nums[r] === nums[r-1]) r--;\n                l++; r--;\n            }\n        }\n    }\n    return out;\n};",
    20: "/**\n * @param {string} s\n * @return {boolean}\n */\nvar isValid = function(s) {\n    const pair = {')':'(',']':'[','}':'{'};\n    const st = [];\n    for (const c of s) {\n        if (pair[c]) {\n            if (!st.length || st.pop() !== pair[c]) return false;\n        } else st.push(c);\n    }\n    return st.length === 0;\n};",
    53: "/**\n * @param {number[]} nums\n * @return {number}\n */\nvar maxSubArray = function(nums) {\n    let best = nums[0], cur = nums[0];\n    for (let i = 1; i < nums.length; i++) {\n        cur = Math.max(nums[i], cur + nums[i]);\n        best = Math.max(best, cur);\n    }\n    return best;\n};",
}

LANG_MAP = {"python": PY_SOLUTIONS, "java": JAVA_SOLUTIONS, "cpp": CPP_SOLUTIONS, "javascript": JS_SOLUTIONS}


def build(rows: int) -> list[dict]:
    out = []
    for problem in PROBLEMS:
        for lang, solutions in LANG_MAP.items():
            sol = solutions.get(problem["problem_id"], "// TODO")
            out.append({
                "problem_id": problem["problem_id"],
                "title": problem["title"],
                "difficulty": problem["difficulty"],
                "description": problem["description"],
                "examples": problem["examples"],
                "constraints": problem["constraints"],
                "language": lang,
                "solution": sol,
            })
            if len(out) >= rows:
                return out
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("./leetcode_dataset.jsonl"))
    p.add_argument("--rows", type=int, default=48)
    args = p.parse_args()

    rows = build(args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
