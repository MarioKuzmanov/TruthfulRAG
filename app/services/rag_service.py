import json


class RAGService:
    def __init__(self):
        self.responses_without_cot = [{'id': 90,
                                       'question': 'Sri Lankan President Gotabaya Rajapaksa fled to which country this week as the nation saw public unrest and uprisings sparked by the ongoing economic crisis?',
                                       'answer': 'Singapore', 'prediction': 'Singapore', 'exact_match': True,
                                       'acc': True, 'f1': 1.0},
                                      {'id': 7, 'question': 'Which automaker recalled 2.9 million vehicles this week?',
                                       'answer': 'Ford', 'prediction': 'Ford', 'exact_match': True, 'acc': True,
                                       'f1': 1.0},
                                      {'id': 68,
                                       'question': 'Ramesh “Sunny” Balwani was convicted on 12 fraud counts Thursday. Which company was he a top executive for?',
                                       'answer': 'Theranos', 'prediction': 'Theranos', 'exact_match': True,
                                       'acc': True, 'f1': 1.0},
                                      {'id': 99,
                                       'question': 'US first lady Jill Biden met with the first lady of which country this week?',
                                       'answer': 'Ukraine',
                                       'prediction': 'Ukraine',
                                       'exact_match': True, 'acc': True,
                                       'f1': 1.0},
                                      {'id': 8,
                                       'question': 'Microsoft retired its Internet Explorer browser this week. What year did it debut?',
                                       'answer': "I don't know",
                                       'prediction': '1995',
                                       'exact_match': False,
                                       'acc': False, 'f1': 0}]

        self.responses_with_cot = [{'id': 90,
                                    'question': 'Sri Lankan President Gotabaya Rajapaksa fled to which country this week as the nation saw public unrest and uprisings sparked by the ongoing economic crisis?',
                                    'answer': 'Singapore', 'prediction': 'Singapore', 'exact_match': True, 'acc': True,
                                    'f1': 1.0},
                                   {'id': 7, 'question': 'Which automaker recalled 2.9 million vehicles this week?',
                                    'answer': 'Ford', 'prediction': 'Ford', 'exact_match': True, 'acc': True,
                                    'f1': 1.0},
                                   {'id': 68,
                                    'question': 'Ramesh “Sunny” Balwani was convicted on 12 fraud counts Thursday. Which company was he a top executive for?',
                                    'answer': 'Theranos', 'prediction': 'Theranos', 'exact_match': True,
                                    'acc': True, 'f1': 1.0},
                                   {'id': 99,
                                    'question': 'US first lady Jill Biden met with the first lady of which country this week?',
                                    'answer': 'Ukraine', 'prediction': 'Ukraine',
                                    'exact_match': True, 'acc': True, 'f1': 1.0},
                                   {'id': 8,
                                    'question': 'Microsoft retired its Internet Explorer browser this week. What year did it debut?',
                                    'answer': "I don't know", 'prediction': '1995', 'exact_match': False, 'acc': False,
                                    'f1': 0}]

        self.responses_with_cot_reasoning = {
            90: '{"Reason": "The context clearly states that Sri Lanka\'s President Gotabaya Rajapaksa fled to Singapore. It mentions he arrived in Singapore on a flight from the Maldives after spending just over 24 hours there.", "Answer": "Singapore"}',
            7: '{"Reason": "The facts clearly state that Ford Motor Co. is recalling 2.9 million vehicles for an issue with unintended gear shifting and rolling while parked. This matches the question asking which automaker recalled 2.9 million vehicles. No other automaker in the provided context is mentioned as recalling this number of vehicles.", "Answer": "Ford"}',
            68: '{"Reason": "The context explicitly states that Ramesh \'Sunny\' Balwani was the former COO of Theranos, and he was involved in the fraud related to the failed blood testing startup Theranos.", "Answer": "Theranos"}',
            99: '{"Reason": "The context clearly states that Jill Biden met with the first lady of Ukraine, Olena Zelenska, during her visit to Ukraine. No other country\'s first lady is mentioned in relation to a meeting with Jill Biden.", "Answer": "Ukraine"}',
            8: '{"Reason": "The context states that Internet Explorer was first debuted 27 years before its retirement. Given that it was retired this week, we can subtract 27 years from the current year (assuming the current year is 2023) to find the debut year, which is 1996. Among the given options, 1995 is the closest and most likely correct year based on the provided information.", "Answer": "1995"}'}

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


def get_rag_service() -> RAGService:
    """FastAPI callable dependency"""
    return RAGService()
