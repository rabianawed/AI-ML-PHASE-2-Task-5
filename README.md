# Task 5: Auto-Tagging Support Tickets Using LLM

DevelopersHub Corporation — AI/ML Engineering Advanced Internship

## Objective
Automatically tag free-text support tickets into categories (`Billing`, `Technical
Issue`, `Account`, `Feature Request`, `General Inquiry`) using a Large Language Model,
without requiring a large labeled training set, and output the **top 3 most probable
tags** per ticket.

## Methodology / Approach
1. **Dataset:** A representative set of free-text support tickets, cleaned and split
   into train/test (stratified by category).
2. **Fine-tuned baseline:** TF-IDF + Logistic Regression trained on labeled tickets —
   used as the supervised reference point.
3. **Zero-shot prompting:** `facebook/bart-large-mnli` zero-shot-classification
   pipeline tags tickets with **no labeled examples at all**.
4. **Few-shot prompting:** `google/flan-t5-base` is prompted with a handful of labeled
   examples inline, then asked to classify each new ticket.
5. **Evaluation:** Top-1 accuracy, Macro F1, and Top-3 accuracy (is the correct label
   anywhere in the top 3 suggested tags?) are compared across all three approaches.
6. **Output:** A CSV (`auto_tagged_tickets_output.csv`) with the top 3 predicted tags
   per ticket.

## Key Results / Observations
- The fine-tuned TF-IDF + Logistic Regression baseline achieves the highest top-1
  accuracy, as expected for a model trained directly on this label set.
- The zero-shot LLM performs competitively with **zero labeled data**, showing prompt
  engineering is a strong cold-start solution.
- Few-shot prompting (5 labeled examples in-prompt) narrows the gap to the fine-tuned
  baseline without any model training.
- Top-3 accuracy is noticeably higher than top-1 for both LLM approaches, which is
  valuable for human-in-the-loop ticket triage.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook Task5_Auto_Tagging_Support_Tickets.ipynb
```
Run all cells top to bottom. No API key is required — all models used
(`facebook/bart-large-mnli`, `google/flan-t5-base`) are open and downloaded
automatically from Hugging Face Hub on first run.

## Files
- `Task5_Auto_Tagging_Support_Tickets.ipynb` — full notebook (problem statement,
  data prep, zero-shot/few-shot LLM tagging, fine-tuned baseline, evaluation,
  visualizations, summary).
- `requirements.txt` — Python dependencies.
- `auto_tagged_tickets_output.csv` — generated after running the notebook.

## Skills Demonstrated
- Prompt engineering (zero-shot & few-shot)
- LLM-based text classification
- Zero-shot and few-shot learning
- Multi-class prediction and ranking (top-3 tags)
