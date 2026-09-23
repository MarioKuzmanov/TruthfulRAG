from transformers import AutoTokenizer

SYSTEM_PROMPTS = {}
SYSTEM_PROMPTS["normal"] = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."
SYSTEM_PROMPTS["qa"] = "You are an expert in retrieval QA. Please respond with the exact answer only. Dont be verbose or provide extra information."
SYSTEM_PROMPTS["extract"] = "You are a precise and reliable information extractor. Your sole task is to extract relevant information from the given context strictly according to the instructions. You must not add, modify, or infer any information that is not explicitly stated in the context."
SYSTEM_PROMPTS["qa-cot"] = "You are an expert in retrieval QA and Chain of Thought reasoning. Provide your reasoning steps followed by a precise and direct answer. Avoiding any unnecessary explanations or verbosity."

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "location", "event"]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following **3** parts of information:
- **entity_name**: Name of the entity, use same language as input text. If English, capitalized the name.
- **entity_type**: One of the following types: [{entity_types}]
- **entity_description**: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>){record_delimiter}

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following **5** parts of information:
- **source_entity**: name of the source entity, as identified in step 1
- **target_entity**: name of the target entity, as identified in step 1
- **relationship_description**: explanation as to why you think the source entity and the target entity are related to each other
- **relationship_strength**: a numeric score indicating strength of the relationship between the source entity and target entity
- **relationship_keywords**: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>){record_delimiter}

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>){record_delimiter}

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
--entities--
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
--relationships--
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
--entities--
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
--relationships--
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
--entities--
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
--relationships--
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS["summarize_entity_descriptions"] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS["entiti_continue_extraction"] = """MANY entities were missed in the last extraction.  Add them below using the same format:"""

PROMPTS["entiti_continue_extraction_mini"] = """MANY entities were missed in the last extraction.
After summarizing with all the information previously extracted, compared to the original text, it was noticed that the following information was mainly omitted:
{omit}

The types of entities that need to be added can be obtained from Entity_types,
or you can add them yourself.

Entity_types: {entity_types}


Add them below using the same format:
"""

PROMPTS["truthfulrag_query2kwd"] = """---Role---

You are a helpful assistant tasked with identifying both **answer_type_keywords** and **entities_from_query** in the user's query.

---Goal---

Given the query, list both **answer_type_keywords** and **entities_from_query**.
**answer_type_keywords** focus on the type of the answer to the certain query, while **entities_from_query** focus on specific entities, details, or concrete terms.
The **answer_type_keywords** must be selected from Answer type pool.
This pool is in the form of a dictionary, where the key represents the Type you should choose from and the value represents the example samples.

---Instructions---

- Attention: You must output the keywords in JSON format directly!
- The JSON should have three keys:
  - "answer_type_keywords" for the types of the answer. In this list, the types with the highest likelihood should be placed at the forefront. No more than 3.
  - "entities_from_query" for specific entities or details. It must be extracted from the query.
######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
Answer type pool: {{
 'PERSONAL LIFE': ['FAMILY TIME', 'HOME MAINTENANCE'],
 'STRATEGY': ['MARKETING PLAN', 'BUSINESS EXPANSION'],
 'SERVICE FACILITATION': ['ONLINE SUPPORT', 'CUSTOMER SERVICE TRAINING'],
 'PERSON': ['JANE DOE', 'JOHN SMITH'],
 'FOOD': ['PASTA', 'SUSHI'],
 'EMOTION': ['HAPPINESS', 'ANGER'],
 'PERSONAL EXPERIENCE': ['TRAVEL ABROAD', 'STUDYING ABROAD'],
 'INTERACTION': ['TEAM MEETING', 'NETWORKING EVENT'],
 'BEVERAGE': ['COFFEE', 'TEA'],
 'PLAN': ['ANNUAL BUDGET', 'PROJECT TIMELINE'],
 'GEO': ['NEW YORK CITY', 'SOUTH AFRICA'],
 'GEAR': ['CAMPING TENT', 'CYCLING HELMET'],
 'EMOJI': ['🎉', '🚀'],
 'BEHAVIOR': ['POSITIVE FEEDBACK', 'NEGATIVE CRITICISM'],
 'TONE': ['FORMAL', 'INFORMAL'],
 'LOCATION': ['DOWNTOWN', 'SUBURBS']
}}
################
Output:
{{
  "answer_type_keywords": ["STRATEGY","PERSONAL LIFE"],
  "entities_from_query": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "When was SpaceX's first rocket launch?"
Answer type pool: {{
 'DATE AND TIME': ['2023-10-10 10:00', 'THIS AFTERNOON'],
 'ORGANIZATION': ['GLOBAL INITIATIVES CORPORATION', 'LOCAL COMMUNITY CENTER'],
 'PERSONAL LIFE': ['DAILY EXERCISE ROUTINE', 'FAMILY VACATION PLANNING'],
 'STRATEGY': ['NEW PRODUCT LAUNCH', 'YEAR-END SALES BOOST'],
 'SERVICE FACILITATION': ['REMOTE IT SUPPORT', 'ON-SITE TRAINING SESSIONS'],
 'PERSON': ['ALEXANDER HAMILTON', 'MARIA CURIE'],
 'FOOD': ['GRILLED SALMON', 'VEGETARIAN BURRITO'],
 'EMOTION': ['EXCITEMENT', 'DISAPPOINTMENT'],
 'PERSONAL EXPERIENCE': ['BIRTHDAY CELEBRATION', 'FIRST MARATHON'],
 'INTERACTION': ['OFFICE WATER COOLER CHAT', 'ONLINE FORUM DEBATE'],
 'BEVERAGE': ['ICED COFFEE', 'GREEN SMOOTHIE'],
 'PLAN': ['WEEKLY MEETING SCHEDULE', 'MONTHLY BUDGET OVERVIEW'],
 'GEO': ['MOUNT EVEREST BASE CAMP', 'THE GREAT BARRIER REEF'],
 'GEAR': ['PROFESSIONAL CAMERA EQUIPMENT', 'OUTDOOR HIKING GEAR'],
 'EMOJI': ['📅', '⏰'],
 'BEHAVIOR': ['PUNCTUALITY', 'HONESTY'],
 'TONE': ['CONFIDENTIAL', 'SATIRICAL'],
 'LOCATION': ['CENTRAL PARK', 'DOWNTOWN LIBRARY']
}}

################
Output:
{{
  "answer_type_keywords": ["DATE AND TIME", "ORGANIZATION", "PLAN"],
  "entities_from_query": ["SpaceX", "Rocket launch", "Aerospace", "Power Recovery"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
Answer type pool: {{
 'PERSONAL LIFE': ['MANAGING WORK-LIFE BALANCE', 'HOME IMPROVEMENT PROJECTS'],
 'STRATEGY': ['MARKETING STRATEGIES FOR Q4', 'EXPANDING INTO NEW MARKETS'],
 'SERVICE FACILITATION': ['CUSTOMER SATISFACTION SURVEYS', 'STAFF RETENTION PROGRAMS'],
 'PERSON': ['ALBERT EINSTEIN', 'MARIA CALLAS'],
 'FOOD': ['PAN-FRIED STEAK', 'POACHED EGGS'],
 'EMOTION': ['OVERWHELM', 'CONTENTMENT'],
 'PERSONAL EXPERIENCE': ['LIVING ABROAD', 'STARTING A NEW JOB'],
 'INTERACTION': ['SOCIAL MEDIA ENGAGEMENT', 'PUBLIC SPEAKING'],
 'BEVERAGE': ['CAPPUCCINO', 'MATCHA LATTE'],
 'PLAN': ['ANNUAL FITNESS GOALS', 'QUARTERLY BUSINESS REVIEW'],
 'GEO': ['THE AMAZON RAINFOREST', 'THE GRAND CANYON'],
 'GEAR': ['SURFING ESSENTIALS', 'CYCLING ACCESSORIES'],
 'EMOJI': ['💻', '📱'],
 'BEHAVIOR': ['TEAMWORK', 'LEADERSHIP'],
 'TONE': ['FORMAL MEETING', 'CASUAL CONVERSATION'],
 'LOCATION': ['URBAN CITY CENTER', 'RURAL COUNTRYSIDE']
}}

################
Output:
{{
  "answer_type_keywords": ["STRATEGY", "PERSON"],
  "entities_from_query": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
Example 4:

Query: "Where is the capital of the United States?"
Answer type pool: {{
 'ORGANIZATION': ['GREENPEACE', 'RED CROSS'],
 'PERSONAL LIFE': ['DAILY WORKOUT', 'HOME COOKING'],
 'STRATEGY': ['FINANCIAL INVESTMENT', 'BUSINESS EXPANSION'],
 'SERVICE FACILITATION': ['ONLINE SUPPORT', 'CUSTOMER SERVICE TRAINING'],
 'PERSON': ['ALBERTA SMITH', 'BENJAMIN JONES'],
 'FOOD': ['PASTA CARBONARA', 'SUSHI PLATTER'],
 'EMOTION': ['HAPPINESS', 'SADNESS'],
 'PERSONAL EXPERIENCE': ['TRAVEL ADVENTURE', 'BOOK CLUB'],
 'INTERACTION': ['TEAM BUILDING', 'NETWORKING MEETUP'],
 'BEVERAGE': ['LATTE', 'GREEN TEA'],
 'PLAN': ['WEIGHT LOSS', 'CAREER DEVELOPMENT'],
 'GEO': ['PARIS', 'NEW YORK'],
 'GEAR': ['CAMERA', 'HEADPHONES'],
 'EMOJI': ['🏢', '🌍'],
 'BEHAVIOR': ['POSITIVE THINKING', 'STRESS MANAGEMENT'],
 'TONE': ['FRIENDLY', 'PROFESSIONAL'],
 'LOCATION': ['DOWNTOWN', 'SUBURBS']
}}
################
Output:
{{
  "answer_type_keywords": ["LOCATION"],
  "entities_from_query": ["capital of the United States", "Washington", "New York"]
}}
#############################

-Real Data-
######################
Query: {query}
Answer type pool: {TYPE_POOL}
######################
Output(Only type **keywords** and **entities**! Don't include Query and Answer type pool!):
"""

PROMPTS["entiti_if_loop_extraction"] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added."""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""


class PromptGenerator:
    def __init__(self, llm_type, task:str="normal", tokenizer=None):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        if task=="qa":
            self.system_prompt = SYSTEM_PROMPTS["qa"]
        elif task=="extract":
            self.system_prompt = SYSTEM_PROMPTS["extract"]
        elif task=="facts":
            self.system_prompt = SYSTEM_PROMPTS["facts"]
        elif task=="qa-cot":
            self.system_prompt = SYSTEM_PROMPTS["qa-cot"]
        else:
            self.system_prompt = SYSTEM_PROMPTS["normal"]

    def _generate_prompt(self, user_prompt):
        return user_prompt
    
    def generate_context_directly_prompt(self, user_query):
        prompt = """
        Generate a background document from Wikipedia to answer the given question:
        {question}. Keep the length of the document around 100 words
        """
        return self._generate_prompt(prompt.format(question=user_query))

    def generate_context_by_factual_knowledge(self, user_query, factual_knowledge):
        prompt = """
        Given the following question and a set of factual knowledge, generate a background document from Wikipedia that can answer the given question. Keep the length of the document around 100 words.
        Question: {question}
        Factual Knowledge: {factual_knowledge}
        Background Document:
        """
        return self._generate_prompt(prompt.format(question=user_query, factual_knowledge=factual_knowledge))

    def generate_factual_knowledge(self, user_query):
        prompt = """
        Task Description:
        You are an expert in problem analysis. When a user presents a question, your task is to identify the factual knowledge required to answer the question. 
        Please list the relevant facts in a clear and structured manner.

        Instructions:
            Analyze the question carefully.
            Identify key areas of knowledge that are crucial for answering the question.
            Provide a brief explanation of why each area is necessary and follow the Example:

        Question:
        Who invented the theory of general relativity?

        Answer:
        To answer this question, the following areas of knowledge are required:
            1. The theory of general relativity describes gravity as a curvature of spacetime caused by mass and energy.
            2. The theory was developed by Albert Einstein in 1915.
            3. Albert Einstein is a German-born theoretical physicist who also developed the equation E = mc², which expresses the equivalence of energy and mass.

        Now, please analyze the following question:

        Question:
        {question}

        Answer:
            1. your first explanation
            2. your second explanation
            3. continue as needed
        """
        return self._generate_prompt(prompt.format(question=user_query))

    def generate_context_extract(self, user_context):
        prompt = """
        Task Description:  
        Extract factual statements based on the given context. Each statement must be concise, accurate, and fully faithful to the information provided in the context. Avoid interpretations, opinions, or assumptions.
        Example:  
        
        Context:  
        The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. It was named after the engineer Gustave Eiffel, whose company designed and built the structure. The tower was completed in 1889 and served as the entrance arch for the 1889 World's Fair.  
        
        Answer:
        The following factual Statements:  
        1. The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.  
        2. The Eiffel Tower was named after the engineer Gustave Eiffel.  
        3. Gustave Eiffel's company designed and built the Eiffel Tower.  
        4. The Eiffel Tower was completed in 1889.  
        5. The Eiffel Tower served as the entrance arch for the 1889 World's Fair.

        Now, please extract the following context:
        Context:  
        {context} 

        Answer:
        1. Your first factual statement
        2. Your second factual statement  
        3. Continue as needed
        """
        return self._generate_prompt(prompt.format(context=user_context))

    def generate_qa_prompt(self, context, question, options = None, facts = None):
        normal_w_facts_prompt = """
-Task Description-
Given facts, a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

-Steps-
1. Analyze the **Question** carefully.
2. Use the **Facts** to provide a clear and accurate answer to the question.
3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.

######################
-Example-
######################
Question:  
Which element has the highest electronegativity?  

Facts:  
Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

Context:  
The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  
#############
Answer: Fluorine

######################
-Real Data-
######################
Question:
{question}

Facts:
{facts}

Context:
{context}
#############
Answer:
"""
        choices_w_facts_prompt = """
-Task Description-
Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

-Steps-
1. Analyze the **Question** and the **Options**.
2. Use the **Facts** to select the most accurate answer from the **Options**.
3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.
4. Please directly answer the option you want to choose. No modification is allowed.

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
Answer: Fluorine

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
Answer:
"""
        normal_wo_facts_prompt = """
-Task Description-
Given a question and a context, your task is to provide a clear and accurate answer. You should only provide the answer based on the context.

-Steps-
1. Analyze the **Question** carefully.
2. Refer to the **Context** to provide a clear and accurate answer to the question.

######################
-Example-
######################
Question:  
Which element has the highest electronegativity?  

Context:  
The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  
#############
Answer: Fluorine

######################
-Real Data-
######################
Question:
{question}

Context:
{context}
#############
Answer:
"""
        choices_wo_facts_prompt = """
-Task Description-
Given a question, a context, and options, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the context and options.
        
-Steps-
1. Analyze the **Question** and the **Options**.
2. Refer to the **Context** to select the most accurate answer from the **Options**.

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
Answer: Fluorine

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
Answer:
"""
        if options is None:
            if facts is None:
                return normal_wo_facts_prompt.format(question=question, context=context)
            else:
                return normal_w_facts_prompt.format(question=question, context=context, facts=facts)
        else:
            if facts is None:
                return choices_wo_facts_prompt.format(question=question, context=context, options=options)
            else:
                return choices_w_facts_prompt.format(question=question, context=context, options=options, facts=facts)

    def generate_qa_prompt_normal_cot(self, context, question, options = None, facts = None, include_context=True):
        normal_w_facts_prompt = """
-Task Description-
Given facts, a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

-Steps-
1. Analyze the **Question** carefully.
2. Use the **Facts** to provide a clear and accurate answer to the question.
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
#############
CoT-Answer:
"""
        choice_w_facts_prompt = """
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
"""
        normal_wo_facts_prompt = """
-Task Description-
Given a question and a context, your task is to provide a clear and accurate answer. You should only provide the answer based on the context.

-Steps-
1. Analyze the **Question** carefully.
2. Refer to the **Context** to provide a clear and accurate answer to the question.
3. Please return in JSON format: {{"Reason": "(reason)", "Answer": "(answer)"}}

######################
-Example-
######################
Question:  
Which element has the highest electronegativity?  

Context:  
The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

Please return in JSON format.
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
#############
CoT-Answer:
"""
        choice_wo_facts_prompt = """
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
"""
        normal_wo_context_prompt ="""
-Task Description-
Given facts and a question, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts.

-Steps-
1. Analyze the **Question** carefully.
2. Use the **Facts** to provide a clear and accurate answer to the question.
3. Please return in JSON format: {{"Reason": "(reason)", "Answer": "(answer)"}}

######################
-Example-
######################
Question:  
Which element has the highest electronegativity?  

Facts:  
Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

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

#############
CoT-Answer:
"""
        choice_wo_context_prompt ="""
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
"""
        # For the ablation with facts only
        if not include_context:
            facts = facts or "" # if no facts pass filtering just provide an empty string
            if options is None:
                return normal_wo_context_prompt.format(question=question, facts=facts)
            else:
                return choice_wo_context_prompt.format(question=question, facts=facts, options=options)
        # Standard prompt options
        if options is None:
            if facts is None:
                return normal_wo_facts_prompt.format(question=question, context=context)
            else:
                return normal_w_facts_prompt.format(question=question, facts=facts, context=context)
        else:
            if facts is None:
                return choice_wo_facts_prompt.format(question=question, context=context, options=options)
            else:
                return choice_w_facts_prompt.format(question=question, facts=facts, context=context, options=options)
