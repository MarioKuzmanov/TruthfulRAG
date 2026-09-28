import unittest
import requests


class TestEndpoints(unittest.TestCase):
    """
    tests that all API endpoints are working as expected
    """

    def test_apis(self):
        shared_item = requests.post("http://localhost:8000/api/setup/select-shared-item/")

        p1 = requests.get("http://localhost:8000/api/prompts/wo-cot-wo-facts")
        p2 = requests.get("http://localhost:8000/api/prompts/cot-wo-facts")
        p3 = requests.get("http://localhost:8000/api/prompts/cot-facts")
        p4 = requests.get("http://localhost:8000/api/prompts/cot-facts-wo-context")
        p5 = requests.get("http://localhost:8000/api/prompts/rag-triples-direct-path-extraction")
        p6 = requests.get("http://localhost:8000/api/prompts/truthfulrag-gliner-raw-predicate-extraction")
        p7 = requests.get("http://localhost:8000/api/prompts/truthfulrag-original-entity-extraction")
        p8 = requests.get("http://localhost:8000/api/prompts/truthfulrag-original-query-to-keywords")

        self.assertEqual(shared_item.status_code, 200, shared_item.text)

        self.assertEqual(p1.status_code, 200, p1.text)
        self.assertEqual(p2.status_code, 200, p2.text)
        self.assertEqual(p3.status_code, 200, p3.text)
        self.assertEqual(p4.status_code, 200, p4.text)
        self.assertEqual(p5.status_code, 200, p5.text)
        self.assertEqual(p6.status_code, 200, p6.text)
        self.assertEqual(p7.status_code, 200, p7.text)
        self.assertEqual(p8.status_code, 200, p8.text)

        m1 = requests.get("http://localhost:8000/api/llm/response-wo-cot")
        m2 = requests.get("http://localhost:8000/api/llm/response-cot")

        self.assertEqual(m1.status_code, 200, m1.text)
        self.assertEqual(m2.status_code, 200, m2.text)

        m3 = requests.get("http://localhost:8000/api/rag/response-wo-cot")
        m4 = requests.get("http://localhost:8000/api/rag/response-cot")

        self.assertEqual(m3.status_code, 200, m3.text)
        self.assertEqual(m4.status_code, 200, m4.text)

        m5 = requests.get("http://localhost:8000/api/rag-triples/view-extracted-paths")
        m6 = requests.get("http://localhost:8000/api/rag-triples/response-wo-context")
        m7 = requests.get("http://localhost:8000/api/rag-triples/response-with-context")

        self.assertEqual(m5.status_code, 200, m5.text)
        self.assertEqual(m6.status_code, 200, m6.text)
        self.assertEqual(m7.status_code, 200, m7.text)

        m8 = requests.get("http://localhost:8000/api/truthfulrag-gliner/filtered-chunks")
        m9 = requests.get("http://localhost:8000/api/truthfulrag-gliner/extracted-entities")
        m10 = requests.get("http://localhost:8000/api/truthfulrag-gliner/extracted-relations")
        m11 = requests.get("http://localhost:8000/api/truthfulrag-gliner/view-extracted-paths")
        m12 = requests.get("http://localhost:8000/api/truthfulrag-gliner/response-wo-context")
        m13 = requests.get("http://localhost:8000/api/truthfulrag-gliner/response-with-context")

        self.assertEqual(m8.status_code, 200, m8.text)
        self.assertEqual(m9.status_code, 200, m9.text)
        self.assertEqual(m10.status_code, 200, m10.text)
        self.assertEqual(m11.status_code, 200, m11.text)
        self.assertEqual(m12.status_code, 200, m12.text)
        self.assertEqual(m13.status_code, 200, m13.text)

        m14 = requests.get("http://localhost:8000/api/truthfulrag-original/filtered-chunks")
        m15 = requests.get("http://localhost:8000/api/truthfulrag-original/extracted-entities")
        m16 = requests.get("http://localhost:8000/api/truthfulrag-original/extracted-relations")
        m17 = requests.get("http://localhost:8000/api/truthfulrag-original/view-extracted-paths")
        m18 = requests.get("http://localhost:8000/api/truthfulrag-original/response-wo-context")
        m19 = requests.get("http://localhost:8000/api/truthfulrag-original/response-with-context")

        self.assertEqual(m14.status_code, 200, m14.text)
        self.assertEqual(m15.status_code, 200, m15.text)
        self.assertEqual(m16.status_code, 200, m16.text)
        self.assertEqual(m17.status_code, 200, m17.text)
        self.assertEqual(m18.status_code, 200, m18.text)
        self.assertEqual(m19.status_code, 200, m19.text)

        health_check = requests.get("http://localhost:8000/health")
        self.assertEqual(health_check.status_code, 200, health_check.text)


if __name__ == '__main__':
    unittest.main()
