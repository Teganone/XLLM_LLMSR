from ..models.openai_model import OpenaiModel

LLM_VERIFT_SYSTEM_PROMPT = 'Whether the "statement" can be deduced from the "evidence" logically, answer with only with True or False, do not output other contents.'

class Verifier:
    def __init__(self, method="llm"):
        self.model = OpenaiModel()
        self.prompt_creator = {
            "LLM":
            "Z3":
        }
        self._load_prompt_templates(self.method)
        
        

    def _load_prompt(method):
        "llm": LLM_VERIFT_SYSTEM_PROMPT
        "z3": LLM_VERIFT_SYSTEM_PROMPT


    def call_llm():
        '''
        call self.model to generate parsing text.
        '''
        pass

    def verify(self, data):
        pass



    
    

    