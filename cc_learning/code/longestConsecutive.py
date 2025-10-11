# 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
#
# 请你设计并实现时间复杂度为 O(n) 的算法解决此问题。



def longestConsecutive(nums: List[int]) -> int:
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[j] - nums[i] == 1:
                nums[i] = nums[j]
                break
    return max(nums) - min(nums)

