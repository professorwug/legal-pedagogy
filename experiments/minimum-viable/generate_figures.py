#!/usr/bin/env python3
"""
Generate figures from the experiment data directly, bypassing Quarto issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set up matplotlib and seaborn styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create figures directory
figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)

# Load data
data_dir = Path('processed_data')

print("Loading experiment data...")

# Load belief distributions
beliefs_df = pd.read_parquet(data_dir / 'belief_distributions.parquet')
print(f"Loaded {len(beliefs_df)} belief measurements")

# Load other files for context
try:
    with open(data_dir / 'extracted_arguments.json', 'r') as f:
        arguments = json.load(f)
    print(f"Loaded {len(arguments)} legal arguments")
except:
    arguments = []

# Character beliefs analysis
print("\n=== Character Belief Distributions ===")
character_beliefs = beliefs_df[beliefs_df['belief_type'] == 'character'].copy()

if not character_beliefs.empty:
    # Create character beliefs plot
    plt.figure(figsize=(12, 8))
    
    # Extract attribute names for better display
    def extract_attribute(question_text):
        """Extract the attribute name from the question text."""
        # Look for the pattern after "appellant's" and before "based on"
        import re
        match = re.search(r"appellant's (.+?) based on", question_text)
        if match:
            return match.group(1)
        return question_text[:50] + "..."
    
    character_beliefs['attribute'] = character_beliefs['question_text'].apply(extract_attribute)
    
    # Create box plot
    sns.boxplot(data=character_beliefs, y='attribute', x='answer')
    plt.title('Character Belief Distributions\n(Justice Kavanaugh Assessment of Appellant)', fontsize=14)
    plt.xlabel('Belief Score (0-1)', fontsize=12)
    plt.ylabel('Character Attributes', fontsize=12)
    plt.xlim(0, 1)
    
    # Add mean values as text
    for i, attr in enumerate(character_beliefs['attribute'].unique()):
        attr_data = character_beliefs[character_beliefs['attribute'] == attr]
        mean_val = attr_data['answer'].mean()
        plt.text(mean_val, i, f'{mean_val:.2f}', 
                ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'character_beliefs.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'character_beliefs.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical summary
    print("\nCharacter Belief Statistics:")
    char_summary = character_beliefs.groupby('attribute')['answer'].agg(['mean', 'std', 'count']).round(3)
    print(char_summary)
    
    # Save summary to CSV
    char_summary.to_csv(figures_dir / 'character_belief_statistics.csv')
    
else:
    print("No character belief data available")

# Distribution analysis
print("\n=== Distribution Analysis ===")
plt.figure(figsize=(12, 5))

# Subplot 1: Overall distribution by type
plt.subplot(1, 2, 1)
sns.histplot(data=beliefs_df, x='answer', hue='belief_type', alpha=0.7, bins=20)
plt.title('Belief Score Distributions by Type')
plt.xlabel('Belief Score (0-1)')
plt.ylabel('Frequency')

# Subplot 2: Character attribute comparison
plt.subplot(1, 2, 2)
if not character_beliefs.empty:
    # Box plot of all attributes
    character_beliefs_plot = character_beliefs.copy()
    character_beliefs_plot['short_attr'] = character_beliefs_plot['attribute'].str[:20]
    sns.boxplot(data=character_beliefs_plot, x='answer', y='short_attr')
    plt.title('Character Attributes Comparison')
    plt.xlabel('Belief Score (0-1)')
    plt.ylabel('Attributes')
else:
    plt.text(0.5, 0.5, 'No character belief data', ha='center', va='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.savefig(figures_dir / 'distribution_analysis.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / 'distribution_analysis.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# Runtime analysis
print("\n=== Runtime Analysis ===")
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.histplot(data=beliefs_df, x='runtime_s', bins=20)
plt.title('Response Time Distribution')
plt.xlabel('Runtime (seconds)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(data=beliefs_df, y='model_name', x='runtime_s')
plt.title('Runtime by Model')
plt.xlabel('Runtime (seconds)')
plt.ylabel('Model')

plt.tight_layout()
plt.savefig(figures_dir / 'runtime_analysis.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / 'runtime_analysis.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# Model performance summary
print("\n=== Model Performance Summary ===")
model_performance = beliefs_df.groupby('model_name').agg({
    'answer': ['count', 'mean', 'std'],
    'runtime_s': ['mean', 'std']
}).round(3)

model_performance.columns = ['Total_Responses', 'Mean_Score', 'Score_StdDev', 'Mean_Runtime', 'Runtime_StdDev']
print(model_performance)

# Save performance summary
model_performance.to_csv(figures_dir / 'model_performance_summary.csv')

print(f"\n✅ All figures saved to {figures_dir}/")
print("📊 Generated files:")
for fig_file in figures_dir.glob('*'):
    print(f"  - {fig_file.name}")

print(f"\n📈 Experiment Summary:")
print(f"  - Total measurements: {len(beliefs_df)}")
print(f"  - Character attributes tested: {character_beliefs['attribute'].nunique() if not character_beliefs.empty else 0}")
print(f"  - Average character score: {character_beliefs['answer'].mean():.3f}" if not character_beliefs.empty else "  - No character data")
print(f"  - Models used: {', '.join(beliefs_df['model_name'].unique())}")