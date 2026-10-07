# Ragas evaluation

Judge: ollama/gemma4:latest · embeddings: intfloat/multilingual-e5-small · 8 questions · 1108s · 20261007-093358 UTC

| Mode | Faithfulness | Answer relevancy | Context precision | Factual correctness | Context recall | Answered |
|---|---|---|---|---|---|---|
| vector | 1.000 | 0.945 | 0.865 | 0.738 | 0.875 | 8/8 |
| graph | 0.945 | 0.896 | 0.559 | 0.615 | 0.750 | 6/8 |
| hybrid | 0.938 | 0.925 | 0.702 | 0.610 | 1.000 | 8/8 |

Per question — factual correctness / context recall:

| Question | Lang | Type | vector | graph | hybrid |
|---|---|---|---|---|---|
| q1: Who is the Managing Director of Ganga Water Systems? | en | lookup | 1.00 / 1.00 | 1.00 / 1.00 | 0.89 / 1.00 |
| q2: Who leads the subsidiary whose steel supplier was put on probation? | en | multi-hop | 0.40 / 0.00 | 0.00 / 0.00 | 0.75 / 1.00 |
| q3: Which vendors connected to Ganga Smart Power are impacted by Clause 7.2? | en | multi-hop | 0.50 / 1.00 | 0.60 / 1.00 | 0.50 / 1.00 |
| q4: Which vendor supplies the most Ganga subsidiaries? | en | aggregation | 0.80 / 1.00 | 0.80 / 1.00 | 0.00 / 1.00 |
| q5: शक्ति स्टील वर्क्स को परिवीक्षा पर क्यों रखा गया? | hi | lookup | 0.86 / 1.00 | 0.91 / 1.00 | 0.91 / 1.00 |
| q6: गंगा स्मार्ट पावर के विक्रेताओं पर कौन सी धारा लागू होती है? | hi | multi-hop | 0.67 / 1.00 | 0.86 / 1.00 | 0.33 / 1.00 |
| q7: Ganga Water Systems ke supplier ko kaun sa clause affect karta hai? | hinglish | multi-hop | 1.00 / 1.00 | 0.75 / 1.00 | 0.83 / 1.00 |
| q8: Group ki 2025 mein revenue kitni thi? | hinglish | lookup | 0.67 / 1.00 | 0.00 / 0.00 | 0.67 / 1.00 |
