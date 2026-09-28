-Task Description-
Given a question, a context, and options, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the context and options.
        
-Steps-
1. Analyze the **Question** and the **Options**.
2. Refer to the **Context** to select the most accurate answer from the **Options**.
3. Please return in JSON format: {{"Reason": "(reason)", "Answer": "(answer)"}}

######################
-Example-
######################
Question:  
Which element has the highest electronegativity?  

Context:  
The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

Options:  
Oxygen  
Chlorine  
Fluorine 
#############
CoT-Answer:
{{"Reason": "According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity.", "Answer": "Fluorine"}}

######################
-Real Data-
######################
Question:
{question}

Context:
{context}

Options:
{options}
#############
CoT-Answer: