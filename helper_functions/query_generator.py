import json
import os
from groq import Groq

class BackstoryDecomposer:
    def __init__(self, llm_client, model_name):
        self.client = llm_client
        self.model = model_name

    def decompose_backstory(self, backstory_text: str, character_name: str) -> list:
        prompt = f"""
        Analyze this backstory for character "{character_name}":
        "{backstory_text}"

        Extract 3-5 atomic, verifiable claims (Temporal, Relationship, Location, Trait).
        For EACH claim, provide 3 search queries:
        1. Keyword search
        2. Descriptive search
        3. Anti-evidence search (checking for contradiction)

        Output JSON format:
        {{
            "claims": [
                {{
                    "text": "Claim description",
                    "type": "CATEGORY",
                    "queries": ["query1", "query2", "query3"]
                }}
            ]
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content).get("claims", [])
        except Exception as e:
            # print(f"Decomposition Error: {e}")
            os.environ["GROQ_KEY_2"] = GROQ_API_KEY_i_2
            self.client = Groq(api_key=os.environ["GROQ_KEY_2"])
            # return [{"text": backstory_text, "type": "GENERAL", "queries": [backstory_text]}]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content).get("claims", [])