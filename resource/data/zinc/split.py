with open("250k_rndm_zinc_drugs_clean_3_canonized.csv") as f:
    lines = f.readlines()
    with open("train.txt", "w") as f1:
        for line in lines[1:200001]:
            f1.write(line.split(',')[0] + "\n")
    with open("valid.txt", "w") as f2:
        for line in lines[200001:225001]:
            f2.write(line.split(',')[0] + "\n")
    with open("test.txt", "w") as f3:
        for line in lines[225001:]:
            f3.write(line.split(',')[0] + "\n")
            
