#time complexity o(n**2)

a = [3,2,5,6,9,7,1]

def insertion_sort(x):
    for i in range(1,len(x)):
        key = x[i]
        j = i-1
        
        while j>=0 and x[j]>key:
            x[j+1] = x[j]
            j -=1
        x[j+1] =key
    return x

print(insertion_sort(a))
#output [1, 2, 3, 5, 6, 7, 9]
