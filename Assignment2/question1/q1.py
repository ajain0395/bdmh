seq_one = "ATGTGTGTCATGCTACGGTCAGGGGTGCATGCTACGTCGTGTCATGTACTG"
seq_two = "ATGTGTGTCATGCTACGGTCAGGGGTGCATGCTACGTCGTGTCATGTACTG"
window = 4
# data = [[(seq_one[i:i + window] != seq_two[j:j + window])
#          for j in range(len(seq_one) - window)]
#         for i in range(len(seq_two) - window)]


seq_one = input("Enter Sequence: ")
seq_two = seq_one
window = int(input("Enter window size: "))

matrix = []
for i in range(len(seq_one)):
    matrix.append([False] *len(seq_two))
#     for j in range(len(seq_two)):
#         matrix[i][j] = False

for i in range(len(seq_one)):
    for j in range(len(seq_two)):
        if(seq_one[i] == seq_two[j]):
            matrix[i][j] = True

tuples = []
for i in range(0,len(seq_one)):
    tuples.append([i,0])
for i in range(0,len(seq_two)):
    tuples.append([0,i])


allseq = []
seq =""

count_repeat = 0
window = 4
for tup in tuples:
    i = tup[0]
    j = tup[1]
    reset = False
    cc = 0
    if( i == j ):
        continue
    while(i < len(seq_one) and j < len(seq_two)):
        if(matrix[i][j] == True):
            cc +=1
            seq += seq_one[i]
        else:
            if(reset == True):
                allseq.append(seq)
            seq = ""
            cc = 0
            reset = False
        if(cc >= window and reset == False):
            reset = True
            count_repeat += 1
        i += 1
        j += 1
    if(len(seq) >= window):
        allseq.append(seq)


print ((set(allseq)))
print ("Repeats: ",len((set(allseq))))




tuples = []
for i in range(0,len(seq_one)):
    tuples.append([i,0])
for i in range(0,len(seq_two)):
    tuples.append([len(seq_one) - 1,i])

allseq = []
seq =""

count_repeat = 0
window = 4
for tup in tuples:
    #print tuples
    i = tup[0]
    j = tup[1]
    reset = False
    cc = 0
    while(i >= 0 and j < len(seq_two)):
        if(matrix[i][j] == True):
            cc +=1
            seq += seq_one[i]
        else:
            if(reset == True):
                allseq.append(seq)
            seq = ""
            cc = 0
            reset = False
        if(cc >= window and reset == False):
            reset = True
            count_repeat += 1
        i -= 1
        j += 1
    if(len(seq) >= window):
        allseq.append(seq)
    
print ((set(allseq)))


print ("Inverse repeats: ",len((set(allseq))))



print ("For the sequence: ",seq_one)