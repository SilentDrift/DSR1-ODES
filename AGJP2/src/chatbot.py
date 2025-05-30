"""DeepSeek ODE Tutor wrapper."""
import pathlib, yaml, os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class DeepSeekODETutor:
    def __init__(self, cfg_path: str | pathlib.Path = None):
        # Find config file relative to project root
        if cfg_path is None:
            project_root = pathlib.Path(__file__).parent.parent
            cfg_path = project_root / "config" / "settings.yaml"
        else:
            cfg_path = pathlib.Path(cfg_path)
        
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
            
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.system_prompt   = cfg["prompts"]["system"]
        self.temperature     = cfg["generation"]["temperature"]
        self.top_p           = cfg["generation"]["top_p"]
        self.max_new_tokens  = cfg["generation"]["max_new_tokens"]
        
        # Handle model path relative to project root
        model_path_str = cfg["model"]["local_path"]
        if not os.path.isabs(model_path_str):
            project_root = pathlib.Path(__file__).parent.parent
            self.model_path = project_root / model_path_str
        else:
            self.model_path = pathlib.Path(model_path_str)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}. Run download_weights.py first.")

        print("🔄 Loading tokenizer & model … (first run can take ~30 s)")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model     = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
            
        self.history = [ {"role": "system", "content": self.system_prompt} ]

    def generate(self, user_msg: str) -> str:
        """Return the assistant's reply and update history."""
        try:
            self.history.append({"role": "user", "content": user_msg})
            prompt = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
            raw = self.pipe(prompt, do_sample=True)[0]["generated_text"]
            answer = raw.split("<|im_start|>assistant", 1)[-1].replace("<|im_end|>", "").strip()
            self.history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(f"❌ {error_msg}")
            return error_msg