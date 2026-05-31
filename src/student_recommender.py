import pandas as pd
import random

df = pd.DataFrame([
    {"task": "Pomodoro Study Session (50/10)", "type": "Deep Work", "state": "Positive"},
    {"task": "Complex Problem Solving / Math Sets", "type": "Deep Work", "state": "Positive"},
    {"task": "Active Recall Flashcards", "type": "Deep Work", "state": "Positive"},
    
    {"task": "Take a 20 min walk outside", "type": "Physical Reset", "state": "Negative"},
    {"task": "Listen to music / podcast", "type": "Mental Reset", "state": "Negative"},
    {"task": "Power nap or guided breathing", "type": "Mental Reset", "state": "Negative"},
    
    {"task": "Organize weekly planner", "type": "Admin", "state": "Neutral"},
    {"task": "Format and review lecture notes", "type": "Maintenance", "state": "Neutral"},
    {"task": "Check university emails", "type": "Admin", "state": "Neutral"}
])

def get_state(preds, w=300):
    if len(preds) > w:
        arr = preds[-w:]
    else:
        arr = preds
        
    c0 = 0
    c1 = 0
    c2 = 0
    
    # count them manually
    for i in arr:
        if i == 0:
            c0 += 1
        elif i == 1:
            c1 += 1
        elif i == 2:
            c2 += 1
            
    # find highest
    if c0 >= c1 and c0 >= c2:
        val = 0
    elif c1 >= c0 and c1 >= c2:
        val = 1
    else:
        val = 2
        
    if val == 0:
        return "Negative"
    if val == 1:
        return "Neutral"
    if val == 2:
        return "Positive"

def rec(state, n=2):
    print("\nState: " + state)
    
    if state == "Positive":
        print("Focus is high.")
    elif state == "Negative":
        print("Fatigue detected.")
    else:
        print("Baseline state.")
        
    # filter df
    temp = df[df["state"] == state]
    
    # picking random ones by converting to list first
    lst = temp.values.tolist()
    random.shuffle(lst)
    
    print("\n--- actions ---")
    count = 0
    for r in lst:
        if count < n:
            print("-> " + str(r[0]) + " (" + str(r[1]) + ")")
            count += 1

# testing
print("running test...")
mock = [2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2]

s = get_state(mock)
rec(s)
