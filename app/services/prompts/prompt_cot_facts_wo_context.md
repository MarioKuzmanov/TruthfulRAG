-Task Description-
Given facts and a question, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts.

-Steps-
1. Analyze the **Question** and the **Options**.
2. Use the **Facts** to select the most accurate answer from the **Options**.
3. Please return in JSON format: {{"Reason": "(reason)", "Answer": "(answer)"}}

######################
-Example-
######################
Question:  
Which element has the highest electronegativity?  

Facts:  
Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

Options:  
Oxygen  
Chlorine  
Fluorine 
#############
CoT-Answer:
{{"Reason": "According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. Therefore, based on the given information, fluorine has the highest electronegativity.", "Answer": "Fluorine"}}

######################
-Real Data-
######################
Question:
{question}

Facts:
{facts}

Options:
{options}
#############
CoT-Answer:
