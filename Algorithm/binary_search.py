# binary search
nums = [17, 20, 26, 31, 44, 54, 55, 77, 93]
target = 55

def binary_search_recur(nums, target):
    """
    递归版本
    返回找到True或者没找到False
    如果要求返回下标不适用递归法
    """
    left = 0
    right = len(nums)-1

    if left < right:
        mid = (left + right) // 2  # 取偏小的那个整数
        if nums[mid] == target:
            return True
        elif nums[mid] < target:
            # 在后半段继续二分查找
            return binary_search_recur(nums[mid+1: ], target)
        else:
            # 在前半段继续二分查找
            return binary_search_recur(nums[:mid], target)

    elif nums[left] == target: # 这个时候left==right
        return True
    else:
        return False


# 非递归版本
def binary_search(nums, target):
    left = 0
    right = len(nums)-1

    while (left < right):
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid+1
        else:
            right = mid-1

    # # 这个时候left==right
    if nums[left] == target:
        return left
    else:
        return -1 # not found



if __name__ == '__main__':

    print(binary_search(nums, 55))
    print(binary_search_recur(nums, 55))
    print(binary_search(nums, 19))
    print(binary_search_recur(nums, 19))
