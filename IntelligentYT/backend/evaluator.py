"""
Final Evaluator for Bachelor Thesis
Supports your current videos
"""

import json
import time
import pandas as pd
from datetime import datetime
from loguru import logger
from rouge_score import rouge_scorer
from bert_score import score as bert_score

class Evaluator:
    def __init__(self):
        self.results = []
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate(self, video_id: str, generated_summary: str, reference_summary: str):
        """Evaluate one video"""
        start = time.time()

        # ROUGE Scores
        rouge_scores = self.scorer.score(reference_summary, generated_summary)

        # BERTScore
        try:
            P, R, F1 = bert_score([generated_summary], [reference_summary], lang="en", verbose=False)
            bert_f1 = F1.mean().item()
        except:
            bert_f1 = None

        result = {
            "video_id": video_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rouge1": round(rouge_scores['rouge1'].fmeasure, 4),
            "rouge2": round(rouge_scores['rouge2'].fmeasure, 4),
            "rougeL": round(rouge_scores['rougeL'].fmeasure, 4),
            "bertscore_f1": round(bert_f1, 4) if bert_f1 else None,
            "generated_length": len(generated_summary.split()),
            "reference_length": len(reference_summary.split())
        }

        self.results.append(result)
        print(f"✅ Evaluated {video_id} | ROUGE-L: {result['rougeL']:.4f} | BERTScore: {result['bertscore_f1']}")
        return result

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv("data/evaluation_results.csv", index=False)
        print("\n📁 Results saved to: data/evaluation_results.csv")
        return df

    def print_summary(self):
        if not self.results:
            print("No results yet.")
            return
        df = pd.DataFrame(self.results)
        print("\n" + "="*70)
        print("📊 EVALUATION SUMMARY (Average)")
        print("="*70)
        print(df[["rouge1", "rouge2", "rougeL", "bertscore_f1"]].mean().round(4))
        print("\nDetailed Results:")
        print(df[["video_id", "rouge1", "rouge2", "rougeL", "bertscore_f1"]].round(4))


# ====================== MAIN EVALUATION ======================
if __name__ == "__main__":
    evaluator = Evaluator()

    print("Starting Evaluation...\n")

    # ================== VIDEO 1 ==================
    gen1 = """TL;DR: Street art and graffiti are powerful forms of urban expression, transforming public spaces into vibrant narratives and challenging traditional notions of art and spatial ownership. They serve as a means of creative expression, social commentary, and community voice, sparking dialogue and sometimes fostering social change. Despite their often-polarized reception, these art forms have become a legitimate contribution to the cultural landscape. 

The evolution of street art has led to its commercialization, transforming what was once a subversive movement, yet it continues to push boundaries and challenge urban cultures. Through its impermanence and public nature, street art engages audiences in real-time, democratizing its consumption and appreciation. Ultimately, street art and graffiti have become a vibrant and influential force in urban cultures, redefining public and private spaces and enriching cities worldwide with their unique perspective."""

    ref1 = """Street art and graffiti are powerful forms of urban expression that transform public spaces into vibrant visual narratives. They challenge traditional ideas of art and ownership, often conveying social and political messages. While sometimes viewed as vandalism, they are increasingly recognized as legitimate cultural contributions. Cities commission murals, but unauthorized works remain controversial. The ephemeral nature of street art makes it unique, and social media has increased its global reach."""

    evaluator.evaluate("ZDvKJcQpf1M", gen1, ref1)

    # ================== VIDEO 2 ==================
    gen2 = """TL;DR: The video follows a monkey as he prepares for an exam, highlighting the importance of effective studying techniques such as active recall, spaced repetition, and honoring the brain's natural ultradian rhythms. By understanding how the brain learns and retains information, individuals can optimize their study habits and improve their performance. The monkey's journey serves as a relatable example of the consequences of poor studying habits and the benefits of adopting a more strategic approach."""

    ref2 = """Effective studying is not about studying longer, but studying smarter. The video uses the story of "Monkey" preparing for an exam to demonstrate key scientific study techniques. The main techniques highlighted are active recall, avoiding the fluency illusion, spaced repetition, ultradian rhythms, and the critical role of sleep for memory consolidation. The key message is that quality and smart techniques matter far more than quantity of study time."""

    evaluator.evaluate("6-TDjSIroRQ", gen2, ref2)

    # Save and show final results
    evaluator.save_results()
    evaluator.print_summary()