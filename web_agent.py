
import sys
print(sys.executable)

from config import API_KEY

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

from custom_multimodal import OpenRouterFallbackMultiModal

models_multimodal = [
    "meta-llama/llama-4-scout:free",
    "meta-llama/llama-4-maverick:free",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "qwen/qwen2.5-vl-32b-instruct:free"
]

mm_llm = OpenRouterFallbackMultiModal(
    models=models_multimodal,
    api_key=API_KEY
)

from custom_textLLM import OpenRouterFallbackTextLLM

models = [  
    "z-ai/glm-4.5-air:free",
    "meta-llama/llama-3.3-70b-instruct:free", 
    "meta-llama/llama-3.3-8b-instruct:free", 
    "mistralai/mistral-7b-instruct:free", 
    "meta-llama/llama-3.2-3b-instruct:free", 
    "meta-llama/llama-3.1-405b-instruct:free" 
]

llm1 = OpenRouterFallbackTextLLM(models=models, api_key=API_KEY)


from generate_yaml import run_resume_pipeline
run_resume_pipeline("SampleResume.pdf",'user_data.pkl')

import pickle
with open("user_data.pkl", "rb") as f:
    user_data = pickle.load(f)

from knowledge_writer import save_knowledge_to_file, extra_knowledge
file_path = save_knowledge_to_file(extra_knowledge)

with open("objective.txt", "r", encoding="utf-8") as f:
    objective = f.read()

from lavague.core import ActionEngine, WorldModel
from lavague.core.agents import WebAgent
from lavague.core.python_engine import PythonEngine
from lavague.drivers.selenium import SeleniumDriver
from lavague.core.memory import ShortTermMemory

def run_web_agent(url, objective, user_data, file_path, mm_llm, llm1, embedding, n_steps=5, headless=True):
   
    # Initialize components
    world_model = WorldModel(mm_llm=mm_llm)
    world_model.add_knowledge(file_path=file_path)

    selenium_driver = SeleniumDriver(headless=headless)
    st_memory = ShortTermMemory()
    current_state, past = st_memory.get_state()
    obs = selenium_driver.get_obs()

    action_engine = ActionEngine(
        llm=llm1,
        driver=selenium_driver,
        embedding=embedding,
        python_engine=PythonEngine(
            driver=selenium_driver,
            llm=llm1,
            embedding=embedding,
            ocr_mm_llm=mm_llm
        )
    )

    agent = WebAgent(world_model, action_engine, n_steps=n_steps)
    agent.get(url)
    agent.logger.new_run()

    try:
        agent.run(objective, user_data=user_data, display=False)
    except Exception as e:
        print(f"⚠️ Agent run failed: {e}")

    # Always return logs, even if run failed
    df_logs = agent.logger.return_pandas()
    return df_logs


