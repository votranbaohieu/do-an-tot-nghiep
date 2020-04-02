newList = []
for i in sentences:
    sent = []
    for word in i.split(" ") :
        if (word not in stop_word) :
            if ("_" in word) or (word.isalpha() == True):
                sent.append(word)
    newList.append(" ".join(sent)) 

