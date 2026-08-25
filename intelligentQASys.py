from llama_cpp import Llama

# ========== 配置区 ==========
# MODEL_PATH = "./models/Qwen3.8-27B-UD-Q4_K_M.gguf"
MODEL_PATH = "./models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
N_GPU_LAYERS = 0       # GPU全部加速；CPU请改为0
N_CTX = 1024            # 上下文窗口大小
MAX_NEW_TOKENS = 600
TEMPERATURE = 0.7
# ============================

def chat_completion(llm: Llama, history: list):
    """
    :param llm: Llama实例对象，由main传入
    :param history: [{"role":"user","content":"xxx"}, {"role":"assistant","content":"xxx"}]
    返回模型回答字符串
    """

    output = llm.create_chat_completion(
        messages=history,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        #stop=["<|im_end|>"],  # Qwen终止符，不同模型终止符不一样
        stop=["<｜end▁of▁sentence｜>"] # deepseek
    )
    ans = output["choices"][0]["message"]["content"]
    return ans

if __name__ == "__main__":
    # 模型加载
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_threads=2,
        verbose=False
    )

    print("==== llama.cpp 本地问答系统 ====")
    print("输入 exit 退出\n")
    chat_history = []
    while True:
        user_input = input("你：")
        user_input = user_input.strip()
        if user_input.lower() == "exit":
            print("退出程序")
            break
        chat_history.append({"role": "user", "content": user_input})
        # 将main内的llm实例作为参数传给chat_completion
        answer = chat_completion(llm, chat_history)
        print(f"AI：{answer}\n")
        chat_history.append({"role": "assistant", "content": answer})