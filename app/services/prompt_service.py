from pathlib import Path


class PromptService:

    def __init__(self):
        self.PROMPTS_ROOT_DIR = Path("app") / "services" / "prompts"

    def prompt_wo_cot_wo_facts(self):
        ## No COT and No Facts
        ## -> applicable to LLM-only and RAG-only without COT
        with open(self.PROMPTS_ROOT_DIR / "prompt_wo_cot_wo_facts.md", "r") as f:
            prompt = f.read()
        return prompt

    def prompt_cot_wo_facts(self):
        ## COT and No Facts
        ## -> applicable to LLM-only and RAG-only with COT
        with open(self.PROMPTS_ROOT_DIR / "prompt_cot_wo_facts.md", "r") as f:
            prompt = f.read()
        return prompt

    def prompt_cot_facts(self):
        ## COT and Facts
        ## -> applicable to RAG + triples, TruthfulRAG-gliner and TruthfulRAG-original
        with open(self.PROMPTS_ROOT_DIR / "prompt_cot_facts.md", "r") as f:
            prompt = f.read()
        return prompt

    def prompt_cot_facts_wo_context(self):
        ## COT and Facts without context
        ## -> applicable to ablations for RAG + triples, TruthfulRAG-gliner and TruthfulRAG-original
        with open(self.PROMPTS_ROOT_DIR / "prompt_cot_facts_wo_context.md", "r") as f:
            prompt = f.read()
        return prompt

    def prompt_original_entity_extraction(self):
        with open(self.PROMPTS_ROOT_DIR / "prompt_original_entity_extraction.md", "r") as f:
            prompt = f.read()
        return prompt

    def prompt_original_query2kwd(self):
        with open(self.PROMPTS_ROOT_DIR / "prompt_original_query2kwd.md", "r") as f:
            prompt = f.read()
        return prompt

    def prompt_direct_path_extraction(self):
        with open(self.PROMPTS_ROOT_DIR / "prompt_direct_path_extraction.md", "r") as f:
            prompt = f.read()
        return prompt

    def prompt_raw_predicate_extraction(self):
        with open(self.PROMPTS_ROOT_DIR / "prompt_raw_predicate_extraction.md", "r") as f:
            prompt = f.read()
        return prompt


def get_prompt_service() -> PromptService:
    return PromptService()
