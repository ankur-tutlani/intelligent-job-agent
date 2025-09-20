from web_agent import run_web_agent,objective,user_data,file_path,mm_llm,llm1,embedding
## specify job posting URL here
URL = "https://www.viaduct.ai/about/careers?gh_jid=4550211101"

df = run_web_agent(
    url=URL,
    objective=objective,
    user_data=user_data,
    file_path=file_path,
    mm_llm=mm_llm,
    llm1=llm1,
    embedding=embedding,
    n_steps =5,
    headless=True
)

print(df.head())  # View logs even if the agent crashed