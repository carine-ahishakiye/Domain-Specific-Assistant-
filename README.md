# Customer Support Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lcKJW8n4BocNdWZuyzL1BXrOGoqV22RY#scrollTo=fyMPrZTfBkq9)

Live Demo: https://carine01-customer-support-assistant.hf.space


## Problem Statement

Customer support teams spend a lot of time answering the same types of questions over and over. Questions about order status, refunds, login issues, and shipping come in constantly and follow very predictable patterns. This project looks at whether a small language model can be trained to handle these kinds of queries automatically, in a way that sounds natural and professional.

I took TinyLlama, a 1.1 billion parameter open source model, and fine-tuned it on real customer support conversations. The goal was to make it respond the way a good support agent would, not just generate generic text.

## Dataset

Source: Bitext Customer Support LLM Chatbot Training Dataset (Hugging Face)
Total size: 26,872 examples
What I used: 5,000 examples sampled randomly
Split: 4,500 for training, 500 for validation

The dataset covers 11 categories including orders, refunds, accounts, payments, shipping, and delivery.

One issue I had to fix was placeholder tokens. The dataset uses things like {{Order Number}} and {{Name}} as templates. Left as-is, these confuse the model during training. I replaced them all with realistic values before doing anything else:

{{Order Number}} becomes ORD-98765
{{Name}} becomes Alex
{{Email}} becomes alex@example.com
{{Refund Amount}} becomes $45.00
{{Tracking Number}} becomes TRK-77889
{{Product Name}} becomes wireless headphones

After that, I formatted each example into an instruction-response template:

```
Below is a customer support query. Respond helpfully and professionally.

### Customer: I need to cancel my order ORD-98765.

### Support Agent: I understand you would like to cancel...
```


## Model and Fine-Tuning

Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Method: LoRA (Low-Rank Adaptation) using the peft library
Training framework: Hugging Face transformers and trl

LoRA keeps most of the model frozen and only trains small adapter matrices inserted into the attention layers. In this case, only 2.25 million parameters were trained out of 1.1 billion total, which is 0.20%. This made it possible to train on a free Google Colab GPU without running out of memory.

The adapters were applied to the q_proj and v_proj layers with rank 16 and alpha 32.


## Experiments

I ran two experiments with different settings to compare which configuration worked better.

| Setting | Experiment 1 | Experiment 2 |
|---|---|---|
| LoRA Rank | 16 | 8 |
| LoRA Alpha | 32 | 16 |
| Learning Rate | 2e-4 | 5e-5 |
| Epochs | 3 | 1 |
| Effective Batch Size | 16 | 16 |
| Trainable Parameters | 2.25M (0.20%) | 1.13M (0.10%) |
| Training Time | 107.6 mins | 37.5 mins |
| Final Training Loss | 0.7984 | 0.89 |
| Best Validation Loss | 0.7177 | 0.8019 |

Experiment 1 performed better. The higher learning rate gave the model more room to adapt to the customer support style, and training for 3 epochs gave it enough time to actually learn the patterns. Experiment 2 was too cautious and did not improve enough in a single epoch.

GPU memory during Experiment 1 on Tesla T4 (15.6 GB total):
Peak usage was around 8.2 GB, which is about 58% less than what full fine-tuning would have required.

---

## Results

Evaluated on 50 validation examples:

| Metric | Base Model | Fine-Tuned | Improvement |
|---|---|---|---|
| ROUGE-1 | 0.3407 | 0.4765 | +39.8% |
| ROUGE-2 | 0.1080 | 0.2382 | +120.5% |
| ROUGE-L | 0.2093 | 0.3394 | +62.2% |
| BLEU | 0.0624 | 0.1396 | +123.9% |
| Perplexity | 6.76 | 2.26 | -66.6% |

All metrics improved by more than 10%.

Here is what the difference looks like on a real query:

Query: Where is my refund? I returned the item two weeks ago.

Base model:
Sure, I would be happy to help you with that. What payment method are you using? ### Customer: I am using PayPal...

The base model loses track of the format and starts generating a fake conversation.

Fine-tuned model:
I am sorry to hear that you are experiencing a delay with your refund. I understand how frustrating it can be to wait, especially after already returning the item. Let me look into this for you right away.

The fine-tuned model stays on track, acknowledges the frustration, and responds the way a real support agent would.

---

## How to Run

On Google Colab :

1. Click the Colab badge at the top of this file
2. Go to Runtime then Change runtime type and select T4 GPU
3. Run all cells from top to bottom
4. Training takes about 107 minutes, total runtime is around 2.5 hours

To run locally:

```bash
git clone https://github.com/carine-ahishakiye/Domain-Specific-Assistant-.git
cd Domain-Specific-Assistant-
pip install transformers datasets peft trl accelerate evaluate rouge-score gradio torch
jupyter notebook notebook.ipynb
```

Requirements: Python 3.8 or higher, a CUDA GPU, and around 20 GB of disk space.

---

## User Interface

The app has three tabs:

Chat: Type any customer query and get a response from the fine-tuned model.

Base vs Fine-Tuned: Both models respond to the same query side by side so you can see exactly what changed after training.

Model Performance: A summary of the training setup and evaluation results.

Try it here: https://carine01-customer-support-assistant.hf.space

---

## Example Conversations

Order cancellation:

Customer: I need to cancel my order ORD-98765.
Agent: I understand you would like to cancel order ORD-98765. I would be happy to help. Could you please verify your account email? Once confirmed I will process the cancellation and you will receive a confirmation within 24 hours.

Account access:

Customer: I cannot log into my account.
Agent: I am sorry to hear you are having trouble. Have you tried resetting your password using the Forgot Password link? If that does not work I can help with account recovery. Just provide the email address associated with your account.

Refund status:

Customer: I cannot track my refund, what is going on?
Agent: I apologize for the confusion. Refunds are typically processed within 5 to 7 business days after we receive the return. Could you share your order number so I can check the exact status for you?

---

## Libraries Used

transformers: model loading and text generation
peft: LoRA implementation
trl: SFTTrainer for supervised fine-tuning
evaluate: ROUGE and BLEU metric calculation
gradio: web interface
Google Colab: free T4 GPU for training
Hugging Face Hub: model hosting and permanent deployment

---

## Acknowledgments

Dataset provided by Bitext. Base model by the TinyLlama team. Training and deployment libraries by Hugging Face.