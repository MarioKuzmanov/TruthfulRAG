-Task Description-
Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

-Steps-
1. Analyze the **Question** and the **Options**.
2. Use the **Facts** to select the most accurate answer from the **Options**.
3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.
4. Please return in JSON format: {{"Reason": "(reason)", "Answer": "(answer)"}}

######################
-Example-
######################
Question:  
Which element has the highest electronegativity?  

Facts:  
Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

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

Facts:
{facts}

Context:
{context}

Options:
{options}
#############
CoT-Answer: