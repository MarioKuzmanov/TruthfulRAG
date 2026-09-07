You will extract subject-predicate-object triples from a source text to build a knowledge graph. A list of entities has already been extracted from this text, and your task is to identify the relationships between these entities.

Your task is to extract triples in the form (subject | predicate | object) where:
- The **subject** must be an entity from the entities list above
- The **object** must be an entity from the entities list above  
- The **predicate** describes the relationship between the subject and object

## Extraction Guidelines

Follow these rules when extracting relationships:

**Accuracy and Faithfulness:**
- Only extract relationships that are explicitly stated or clearly implied in the source text
- Do not infer relationships that go beyond what the text states
- Extract only factual relationships (not opinions, hypotheticals, or negated statements)

**Predicate Quality:**
- Use clear, specific predicates that precisely describe the relationship
- Examples of good predicates: "founded_by", "located_in", "works_for", "invented", "part_of", "owns", "manages", "created", "acquired"
- Avoid vague predicates like "related_to" or "associated_with" unless no more specific relationship can be determined

**Directionality:**
- Ensure the subject-predicate-object order correctly represents the relationship direction
- For example: (Apple | founded_by | Steve Jobs) not (Steve Jobs | founded_by | Apple)

**Thoroughness:**
- Extract ALL meaningful relationships present in the text
- Pay special attention to entities that might otherwise have no connections—Entities may legitimately have no relationship to another supplied entity. Never create a relationship solely to avoid an isolated entity.
- Your goal is to maximize connectivity in the knowledge graph and minimize isolated entities