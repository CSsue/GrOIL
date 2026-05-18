import os
from pathlib import Path
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
OUT_DIR = Path(os.getenv("OUT_DIR", "./out"))

ALL_PATH  = DATA_DIR / "100_contracts_dataset/contracts"
SEED_PATH = DATA_DIR / "10_contracts_dataset/contracts"
OUT_PATH  = OUT_DIR / "contract_diversity.png"

CATEGORY_MAP = {
    "1": "Basic Term Life",        "2": "Whole Life with Dividends",
    "3": "Universal Life",         "4": "Variable Universal Life",
    "5": "Final Expense",          "6": "Group Term Life",
    "7": "Term Life with Riders",  "8": "Joint Survivorship",
    "9": "Return of Premium Term", "10": "Indexed Universal Life",
}
SHORT_NAMES = { 
    "1": "Basic Term", "2": "Whole Life", "3": "Universal Life", "4": "Variable UL", "5": "Final Expense",
    "6": "Group Term", "7": "Term w/ Riders", "8": "Joint Surv.", "9": "Return of Prem.", "10": "Indexed UL"
}

COLORS = ["#F23434", "#FF9F18", "#34820D", "#2BC1B0", "#04414E",
          "#0057FF", "#6A0FBD", "#DC0C7F", "#853A05", "#5D5A5A"]

MARKERS = ['o', 's', 'D', 'X', 'p', 'H', 'P', 'v', 'd', 'h']

plt.rcParams.update({'font.family': 'serif', 'savefig.dpi': 300})

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    records = []
    active_folders = [d for d in ALL_PATH.iterdir() if d.is_dir()]
    print(f"Found {len(active_folders)} active folders: {[d.name for d in active_folders]}")
    
    active_ids = {d.name.replace('CT', '') for d in active_folders}

    for f in SEED_PATH.glob("contract_*_*.txt"):
        cat = f.name.split('_')[1]
        if cat in active_ids:
            records.append({'Category': cat, 'Type': 'Seed', 'text': f.read_text(errors='ignore')})

    for d in active_folders:
        clean_cat = d.name.replace('CT', '') 
        for f in d.glob("*.txt"):
            records.append({'Category': clean_cat, 'Type': 'Variant', 'text': f.read_text(errors='ignore')})

    return pd.DataFrame(records)

# ── Embeddings ────────────────────────────────────────────────────────────────

def embed(texts):
    tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=20000)
    return TruncatedSVD(n_components=50, random_state=42).fit_transform(tfidf.fit_transform(texts))

# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(df, embeddings):
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))

    proj = umap.UMAP(n_components=2, n_neighbors=9, min_dist=0.5,
                     metric='cosine', random_state=42).fit_transform(embeddings)
    df['x'], df['y'] = proj[:, 0], proj[:, 1]

    categories = sorted(df['Category'].unique(), key=int)
    labels = [SHORT_NAMES[c] for c in categories]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [1.6, 1]})
    

    # ── Left: UMAP ────────────────────────────────────────────────────────────
    ax1.grid(linestyle='--', alpha=0.2, color='#CCCCCC', zorder=0)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.tick_params(length=0)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_xlabel("UMAP Component 1", fontsize=11)
    ax1.set_ylabel("UMAP Component 2", fontsize=11)

    handles = []
    for i, cat in enumerate(categories):
        c, m = COLORS[i], MARKERS[i]
        name = CATEGORY_MAP.get(cat, cat)
        v = (df['Category'] == cat) & (df['Type'] == 'Variant')
        s = (df['Category'] == cat) & (df['Type'] == 'Seed')
        sc = ax1.scatter(df.loc[v, 'x'], df.loc[v, 'y'],
                         color=c, marker=m, s=50, edgecolor='w', linewidth=0.5, zorder=3)
        handles.append(sc)
        if s.any():
            ax1.scatter(df.loc[s, 'x'], df.loc[s, 'y'],
                        color='white', marker='*', s=150, edgecolor=c, linewidth=1.75, zorder=5)

    seed_handle = ax1.scatter([], [], color='white', marker='*', s=100,
                              edgecolor='black', linewidth=1.75)

    # ── Right: Boxplot ────────────────────────────────────────────────────────
    data = [df[df['Category'] == c]['word_count'].values for c in categories]
    bp = ax2.boxplot(data, patch_artist=True, vert=False, showfliers=False,
                     medianprops=dict(color='white', linewidth=1.5),
                     boxprops=dict(linewidth=0.8),
                     whiskerprops=dict(linewidth=0.8),
                     capprops=dict(linewidth=0.8))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax2.set_yticks([])
    ax2.set_xlabel("Word Count", fontsize=11)
    ax2.xaxis.set_tick_params(labelsize=8, rotation=0)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(200))
    ax2.grid(axis='x', linestyle='--', alpha=0.3, color='#CCCCCC', zorder=0)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(length=0)

    ax1.annotate("(a)", xy=(0.5, -0.18), xycoords='axes fraction', ha='center', fontsize=11)
    ax2.annotate("(b)", xy=(0.5, -0.18), xycoords='axes fraction', ha='center', fontsize=11)

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_labels = [CATEGORY_MAP[c] for c in categories] + ['Seed contract']
    fig.legend(handles + [seed_handle], legend_labels,
               loc='lower center', ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.9),
               handletextpad=0, columnspacing=0.3)

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    plt.savefig(OUT_PATH, bbox_inches='tight')
    print(f"Saved diversity plot to -> {OUT_PATH}")
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    embeddings = embed(df['text'].tolist())
    plot(df, embeddings)