from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline, AutoConfig
import time

# env.localModelPath = '../../assets/models/'
# env.allowRemoteModels = False

model_id = "codegen-350M-mono"
path = "../../assets/models/"

# config = AutoConfig.from_pretrained(path + model_id + "config.json")
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
model = OVModelForCausalLM.from_pretrained(path + model_id, ov_config=ov_config, compile=False, trust_remote_code=True, device="CPU")
model.compile()

tokenizer = AutoTokenizer.from_pretrained(path + model_id)
cls_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

st = time.process_time()
outputs = cls_pipe("def fib(n):")
et = time.process_time()

rt = et - st

print(outputs)
print(f"{rt:.2f}s")
