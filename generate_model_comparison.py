
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_comparison():
    # Performance data based on final training runs
    models = ['Baseline (Greedy)', 'Llama-3.2-1B', 'Qwen-2.5-7B-GRPO']
    
    # Metrics
    equity_index = [0.732, 0.840, 0.912]
    avg_reward = [0.548, 0.785, 0.864]
    utility_score = [0.545, 0.712, 0.808]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width, equity_index, width, label='Equity Index (Fairness)', color='#FF9900')
    rects2 = ax.bar(x, avg_reward, width, label='Avg Reward', color='#1a73e8')
    rects3 = ax.bar(x + width, utility_score, width, label='Utility Score', color='#34a853')
    
    ax.set_ylabel('Scores (0-1)')
    ax.set_title('FairRecovery++ Model Comparison: Llama vs Qwen vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/model_comparison.png', dpi=150)
    print("Saved assets/model_comparison.png")

if __name__ == "__main__":
    generate_comparison()
