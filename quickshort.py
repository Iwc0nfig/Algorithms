#time complexity O(nlogn)

def partion(arr,low,high):
    pivot = arr[high]
    i =  low-1
    for j in range(low,high):
        if arr[j] <=pivot:
            i +=1
            arr[i] , arr[j] = arr[j],arr[i]
            
    arr[i+1] , arr[high] = arr[high],arr[i+1]
    return i+1

def quickshort(arr,low,high):
    if low <high:
        pivot_index =partion(arr,low,high)
        print(arr)
        quickshort(arr,low,pivot_index-1)
        quickshort(arr,pivot_index+1,high)
        return arr


a = [3,2,6,7,4,9,5]
b = [3,2,4]
print(len(a))
print(quickshort(a,0,len(a)-1))
