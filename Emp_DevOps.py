class employee:
    def __init__(self,empid,empname,age,ismarried):
        self.empid = empid
        self.empname=empname
        self.age = age
        self.ismmarried = ismarried
emparr = [
    employee("4578", "Chunky", 18, True),
    employee("4579", "Punky ", 19, True),
    employee("4580", "Munky ", 19, False),
    employee("4581", "Lunky ", 20, False),
    employee("4582", "Yunky ", 21, True),

]
print("_____________" + " ___________")
print("Employee Name" + " Employee ID")
print("_____________" + " ___________")

count = 0
for q in emparr:
    print(emparr[count].empname + "       |   " + emparr[count].empid )
    count += 1
