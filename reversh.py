start = 1
end = 20

for i in range(start,end + 1):
    if i > 1:
        for j in range(int(i//2)):
            if j % 2 == i :
                print (i)
                break