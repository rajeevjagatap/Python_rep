from survey import survey
question_prompts = [
    "Rajeev is a male \n a. True \n b.False\n What is your answer: \n",
    "Rajeev is a girl \n a. True \n b.False\n What is your answer: \n",
    "India is Country \n a. True \n b.False\n What is your answer: \n",
]


q1 = [
    survey(question_prompts[0], "a"),
    survey(question_prompts[1], "b"),
    survey(question_prompts[2], "a"),
]

def test(q1):
    print("Enter the Function")
    score = 0
    for f1 in q1:
        print("enter for" + str(score))
        answer2 = input(f1.pass1)
        if answer2 == f1.pass2:
            score += 1
    print("You Score " + str(score))
test(q1)