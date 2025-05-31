# skyGPT

## Purpose:
This project aims to recreate the core architecture behind large language models like GPT, LLaMA, and DeepSeek, as well as foundational LLM techniques such as fine-tuning and distillation — all from scratch. While the resulting model could have some practical applications, it is primarily a personal project to help me develop a deep, hands-on understanding of the key concepts and mechanisms that underpin modern AI. It also provides me with a solid codebase for future experimentation and exploration of my own ideas. However, learning is the ultimate goal.

The project also has resource constraints in both compute and data. As a result, the final model will initially be limited to around 50–100 million parameters and be trained on a diverse corpus (web text, speech transcripts, academic papers, books, code, etc) totaling approximately 15–20 GB. However, the code and architecture is designed to scale if additional resources become available.

In terms of use cases, small (I guess here it's super small) language models offer advantages like their ease of fine-tuning and low compute requirements for both training and inference. This makes them well-suited for local deployment on edge devices for specialized, niche tasks, and is something that could be explored after the conclusion of this project.

Finally, since necessity is the mother of innovation, I am hopeful that working on a model with such resource and parameter constraints will inspire me to experiment with new ideas and techniques that try to overcome those limitations (beyond simply scaling up).

## Progress 
- **Update (31/05/2025)**: The project has officially begun with the completion of the first milestone: reproducing nanoGPT, a super small and simple language model developed by Andrej Karpathy that recreates the architecture from the seminal "Attention Is All You Need" paper. This step took some time, as I was also diving deep into the underlying concepts of transformers and self-attention which are arguably the most critical ideas in modern LLMs. 

- There are still some details that need to be cleaned up like with changing the hyperparameters for number of attention heads and number of attention+ff blocks, but I will patch those in soon.

- With this foundation in place, I'm excited to move on to the next set of goals. I expect each to take around 1–2 weeks, which also allows time for further research and learning of the topics as I go. After every milestone I will also release an additional notebook with observations, comments and extra information that illustrates my learning process. I will release the notebook for this goal soon (just need to clean it up)

## All Goals:
- **Attention Is All You Need**: Build a transformer (decoder) from scratch, inspired by Andrej Karpathy's nano-gpt
- **Closer to modern LLMs**: Scale up the model, identify a suitable training corpus, add features such as visualization tools and auto hyperparameter tuning, and build an interface (CLI or otherwise) to facilitate easy testing and experimentation with hyperparameters and future features.
- **Variations of the attention mechanism**: Implement and test attention mechanisms like: Sliding window attention (with longformer), flash attention, multi-head latent attention (DeepSeek)
- **Build an Encoder**: For embedding and Seq2Seq tasks like translation
- **Parameter efficient fine-tuning**: Prefix Tuning, Prompt Tuning, LoRA/QLoRA, Side-Tuning
- **Distillation**: Distill a larger model into SKY or SKY into a larger model. Experiment with Co-Distillation (Llama)
- **Augmentations Part 1 (standard)**: RLHF, Reasoning, Simple RAG (using sentence embeddings)
- **Augmentations Part 2 (experimental)**: Memorizing Transformers, other Test Time Scaling methods, Google Titans

## How To Run
NOTE: These instructions will change depending on the phase of the project. For now, since the project is still a recreation of nano-gpt on the Tiny Shakespeare dataset, the instructions are just to 


1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd skyGPT
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   cd code
   python nano_sky.py
   ```
