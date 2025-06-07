# ─── granite/client.py ─────────────────────────────────────────────────────────

from ibm_granite import GraniteClient, GraniteModels
from config.settings import GRANITE_API_KEY, GRANITE_ENDPOINT, GRANITE_MODEL_NAME

class GraniteAPI:
    """
    Thin wrapper around IBM Granite 13B Instruct. 
    """
    def __init__(self):
        self.client = GraniteClient(apikey=GRANITE_API_KEY, endpoint=GRANITE_ENDPOINT)
        self.model = self.client.get_model(GraniteModels[GRANITE_MODEL_NAME])

    def generate_text(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        response = self.model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.get("generated_text", "").strip()
