# Goal

This repository is currently a work-in-progress. The focus of this project is to build a question-answering model by fine-tuning a large language model (LLM) on simple and safe $(S^{2})$ text. 
Here by "simple text" we mean the text should be easily understood by someone without a college degree and ideally by middle-school students or by people for whom English is a second-language.
The text should also be "safe" in the sense it should not contain any potentially toxic content, i.e. any obscenities or disparaging comments to any individual or group. 
Our focus on these simple and safe-text is motivated by two factors:

1. We ideally want to create an LLM which can answer questions and always does so in a safe and harmless manner. By performing supervised fine-tuning (SFT) on safe text, we are pushing the LLM in the direction of being harmless.
2. Two recent papers, <a href="https://arxiv.org/abs/2305.07759">TinyStories</a> and <a href="https://arxiv.org/abs/2306.11644">Textbooks Are All You Need</a> show that one can get surprisingly good results by training a relatively small model on a high-quality dataset. In particular, in the TinyStories paper they showed that GPT-2 can produce coherent stories if it is trained on dataset containing words that a typical 3- or 4-year old can understand. Therefore, we expect that training LLMs on simple text may boost their performance and make them competitive with larger models and/or equivalently sized models trained on more text.
3. Finally, we believe fine-tuning a model on simple text may help the LLM produce text which is simpler to understand for people learning English or for students in elementary or middle school.

# Simple and Safe ($S^{2}$) Text

How do we actually determine whether a text is simple? This is certainly a difficult question and its not clear how to define these terms in an objective manner. 
For this work, we decided to use the <a href="https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests">Flesch-Kincaid readability metrics</a> as a simple baseline. Specifically, we use the Flesch reading ease (FRE) and Flesch–Kincaid grade level (FKG) tests to measure the complexity of a text. 
A higher score on FRE indicates that the text is easier to read while the FKG score corresponds roughly to the US grade system.
One benefit of using these two metrics is that they are relatively fast to compute and can be used to quickly filter large datasets. 
We will say a text is "simple" if it achieves an FRE score $\geq60$ and a FKG score $\leq 9$, i.e. it should be an easily readable text and written at a strictly $9^{th}$-grade reading level or less (the FKG score is continuous and can take any value between 9 and 10).

To measure toxicity we use the "unbiased" model from <a href="https://github.com/unitaryai/detoxify">UnitaryAI</a>, which is a fine-tuned RoBERTa-base model. This model assigns a text a score (between 0 and 1) in the following categories: toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit. A score of 1 indicates high-toxicity while a score of 0 indicates that the text is non-toxic. We will say a text is non-toxic, or safe, if it achieves a score of at most 0.1 in all 7 categories.

# Models and Datasets

In this project we primarily use Llama-2 as our base model. Thus far, we have fine-tuned the 7B or 13B parameter models using <a href="https://github.com/artidoro/qlora">QLORA</a>. Although these models are larger than the models studied in TinyStories or "Textbooks Are All You Need", they are small in comparison to state of the art models, such as GPT-3.5 (175B parameters). For our training data, we use this cleaned <a href="https://huggingface.co/datasets/vblagoje/lfqa">version</a> of the original <a href="https://huggingface.co/datasets/eli5">ELI5</a> dataset and the Simple Wikipedia split of the <a href="https://huggingface.co/datasets/wikipedia/viewer/20220301.simple/train">Wikipedia</a> dataset.
To turn the Simple Wikipedia dataset, which just consists of articles scraped from <a href="https://simple.wikipedia.org/wiki/Main_Page">Simple Wikipedia</a> we prompted GPT-3.5 to generate a question whose answer is the first few paragraphs of the Simple Wikipedia article. By adopting this approach, we are quickly able to generate high-quality question/answer pairs without human annotation. More details on our dataset and training procedure will be forthcoming and will be included in the **reports** folder.
