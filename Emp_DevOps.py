#import employee type from functions
from functions import employee


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
