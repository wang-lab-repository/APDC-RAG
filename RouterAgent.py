import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
class RouterAgent:
    def __init__(self, model_path="./router/t5_router_final_model", use_router=True):
        self.use_router = use_router
        if self.use_router:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            self.model.eval()

    def predict_modality(self, query):
        if not self.use_router:
            return "fusion"
        input_text = f"route: {query}" 
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10)
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()
        

        if "vision" in prediction or "visual" in prediction:
            return "visual"

        if "text" in prediction:
            return "text"

        return "fusion"