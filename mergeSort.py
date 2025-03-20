#time complexity o(nlogn)
#append used to add one element 
#extend used to add a list of elements

def mergeSort(arr):
    if len(arr)<=1:
        return arr

    mid = len(arr)//2
    left = arr[:mid]
    right = arr[mid:]

    sortedLeft = mergeSort(left)
    sortedRight = mergeSort(right)

    return merge(sortedLeft,sortedRight)

def merge(left,right):
    result =[]

    i = j = 0

    while len(left) >i and len(right)>j:
        if left[i] <right[j]:
            result.append(left[i])
            i +=1
        else:
            result.append(right[j])
            j +=1

    #We add first the left[i:] because we know that they are bigger numbers than the result and smaller numbers from the right[j:]
    result.extend(left[i:])
    result.extend(right[j:])
    return result


arr = [4,3,7,6,9,-1,2]
mergeSort(arr)
#output [-1, 2, 3, 4, 6, 7, 9]
