import json

class LLMService:
    def __init__(self):
        # the responses are already computed, so we keep the service lightweight and only demo the observed behavior
        self.responses_without_cot = [{'id': 90,
                                       'question': 'Sri Lankan President Gotabaya Rajapaksa fled to which country this week as the nation saw public unrest and uprisings sparked by the ongoing economic crisis?',
                                       'answer': 'Singapore', 'prediction': 'Singapore', 'exact_match': True,
                                       'acc': True,
                                       'f1': 1.0},
                                      {'id': 7, 'question': 'Which automaker recalled 2.9 million vehicles this week?',
                                       'answer': 'Ford', 'prediction': "I don't know", 'exact_match': False,
                                       'acc': False, 'f1': 0},
                                      {'id': 68,
                                       'question': 'Ramesh “Sunny” Balwani was convicted on 12 fraud counts Thursday. Which company was he a top executive for?',
                                       'answer': 'Theranos', 'prediction': 'Theranos',
                                       'exact_match': True, 'acc': True, 'f1': 1.0},
                                      {'id': 99,
                                       'question': 'US first lady Jill Biden met with the first lady of which country this week?',
                                       'answer': 'Ukraine', 'prediction': 'Ukraine', 'exact_match': True, 'acc': True,
                                       'f1': 1.0},
                                      {'id': 8,
                                       'question': 'Microsoft retired its Internet Explorer browser this week. What year did it debut?',
                                       'answer': "I don't know", 'prediction': '1995', 'exact_match': False,
                                       'acc': False, 'f1': 0}
                                      ]

        self.responses_with_cot = [{'id': 90,
                                    'question': 'Sri Lankan President Gotabaya Rajapaksa fled to which country this week as the nation saw public unrest and uprisings sparked by the ongoing economic crisis?',
                                    'answer': 'Singapore',
                                    'prediction': "I don't know",
                                    'exact_match': False,
                                    'acc': False,
                                    'f1': 0},
                                   {'id': 7,
                                    'question': 'Which automaker recalled 2.9 million vehicles this week?',
                                    'answer': 'Ford',
                                    'prediction': "I don't know",
                                    'exact_match': False,
                                    'acc': False,
                                    'f1': 0},
                                   {'id': 68,
                                    'question': 'Ramesh “Sunny” Balwani was convicted on 12 fraud counts Thursday. Which company was he a top executive for?',
                                    'answer': 'Theranos',
                                    'prediction': 'Theranos',
                                    'exact_match': True,
                                    'acc': True,
                                    'f1': 1.0},
                                   {'id': 99,
                                    'question': 'US first lady Jill Biden met with the first lady of which country this week?',
                                    'answer': 'Ukraine',
                                    'prediction': "I don't know",
                                    'exact_match': False,
                                    'acc': False,
                                    'f1': 0},
                                   {'id': 8,
                                    'question': 'Microsoft retired its Internet Explorer browser this week. What year did it debut?',
                                    'answer': "I don't know",
                                    'prediction': '1995',
                                    'exact_match': False,
                                    'acc': False,
                                    'f1': 0}]

        self.responses_with_cot_reasoning = {
            90: '{"Reason": "The question asks about the specific country Sri Lankan President Gotabaya Rajapaksa fled to during public unrest due to an economic crisis. However, the provided facts and context do not contain any information about his whereabouts. Therefore, based on the available information, the correct answer is that the information is not known.", "Answer": "\\"I don\'t know\\""}',
            7: '{"Reason": "The facts and context provided do not contain specific information about any automaker recalling 2.9 million vehicles this week. Therefore, based on the available information, the correct answer is that the identity of the automaker is unknown.", "Answer": "I don\'t know"}',
            68: '{"Reason": "The facts indicate that Ramesh ‘Sunny’ Balwani was convicted on 12 fraud counts. He was a key figure at Theranos, where he served as the company’s chief operating officer and played a significant role in the company’s operations and fraudulent activities.", "Answer": "Theranos"}',
            99: '{"Reason": "The question asks about a meeting between US First Lady Jill Biden and another country\'s first lady. However, no specific information about such a meeting is provided in the given facts or context. Therefore, based on the available information, the correct answer is \'I don\'t know\'.", "Answer": "\\"I don\'t know\\""}',
            8: '{"Reason": "The question asks for the debut year of Internet Explorer, which is directly related to the facts and context provided. Although no specific fact about the debut year is mentioned, the options give us the correct answer. Microsoft\'s Internet Explorer was first released in 1995.", "Answer": "1995"}'}

    def response_without_cot(self, item_id: str) -> dict:
        for response in self.responses_without_cot:
            if str(response["id"]) == item_id:
                return {"predicted_answer": response['prediction'], "acc": 1.0 if response['acc'] else 0.0,
                        "f1": response['f1']}
        return None

    def response_with_cot(self, item_id: str) -> dict:
        for response in self.responses_with_cot:
            if str(response["id"]) == item_id:
                # more controlled extraction is needed but our examples are hardcoded
                reasoning = json.loads(self.responses_with_cot_reasoning[response["id"]])["Reason"]
                return {"predicted_answer": response['prediction'], "acc": 1.0 if response['acc'] else 0.0,
                        "f1": response['f1'], "reasoning": reasoning}
        return None


def get_llm_service() -> LLMService:
    """FastAPI callable dependency"""
    return LLMService()
