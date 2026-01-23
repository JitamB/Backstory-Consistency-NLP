import json
import os
import time
from groq import Groq
from groq import RateLimitError

class StoryVerifier:
    def __init__(self, llm_client, model_name):
        self.client = llm_client
        self.model = model_name

    def verify_backstory(self, claims, retrieval_engine, book_title):
        contradiction_found = False
        all_rationales = []

        for claim in claims:
            evidence = retrieval_engine.search(claim, book_title)
            if not evidence: continue

            evidence_text = "\n".join([f"- {e['chunk_text']} (Pos: {e['relative_position']})" for e in evidence])

            prompt = f"""
            Task: Check Consistency.
            Claim: "{claim['text']}"
            Evidence from Book:
            {evidence_text}

            Does the evidence IMPOSSIBLY CONTRADICT the claim?

            Return a JSON object with this exact format:
            {{
                "verdict": "SUPPORT" or "CONTRADICT" or "NEUTRAL",
                "confidence": 0.0 to 1.0,
                "rationale": "One sentence explanation"
            }}
            """
            retries = 2
            for i in range(retries):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a strict fact-checker. Output JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.0
                    ).choices[0].message.content

                    result = json.loads(resp)

                    verdict = result.get("verdict", "NEUTRAL").upper()
                    confidence = float(result.get("confidence", 0.0))
                    rationale = result.get("rationale", "")

                    if verdict == "CONTRADICT" and confidence >= 0.4:
                        contradiction_found = True
                    all_rationales.append(f"{rationale}")
                    time.sleep(3)
                    break # Break retry loop if successful
                except RateLimitError as e:
                    # print(f"Verification Rate Limit Error for claim '{claim['text']}' (attempt {i+1}/{retries}): {e}")
                    print("Switching api key")
                    time.sleep(2 ** i + 1) # Exponential backoff with a base of 1 second
                    os.environ["GROQ_KEY_2"] = GROQ_API_KEY_i_2
                    self.client = Groq(api_key=os.environ["GROQ_KEY_2"])
                    continue
                except Exception as e:
                    print(f"Verification Error for claim '{claim['text']}': {e}")
                    break # Break retry loop for other errors

        if contradiction_found:
            return 0, " | ".join(all_rationales)
        return 1, " | ".join(all_rationales)