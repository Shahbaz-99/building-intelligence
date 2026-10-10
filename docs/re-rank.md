Semantic + keyword → fused list
                       ↓
                 keep top 20       ← candidate limit
                       ↓
                 send first 10     ← send limit
                       ↓
                 reranker scores 10
                       ↓
                 return best 5     ← return limit