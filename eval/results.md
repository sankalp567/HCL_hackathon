# Evaluation results

Backend: `http://127.0.0.1:8765` · questions: 31 · run at 2026-10-27 13:13

| Score | Result |
|---|---|
| Expected outcome (answered / not found / out of scope) | 31/31 (100%) |
| Numeric answers with the expected numbers | 20/21 (95%) |
| Descriptive answers with the expected term | 4/4 (100%) |
| Numeric claims whose numbers are on a cited PDF page | 35/35 (100%) |
| Latency p50 / p95 | 0.0 s / 4.5 s |

Citations are checked against each page's own text layer, independent of facts.db.

| ID | Type | Question | Status | Numbers | Terms | Cited numbers on page | Answered by | Seconds |
|---|---|---|---|---|---|---|---|---|
| L01 | lookup | What was TCS's revenue from operations in FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L02 | lookup | What was Infosys's revenue in FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L03 | lookup | What was HCLTech's revenue in FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L04 | lookup | What was TCS's revenue in FY2024-25? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L05 | lookup | What was Infosys's net profit in FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L06 | lookup | What was HCLTech's net profit in FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L07 | lookup | What was TCS's net profit in FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L08 | lookup | What was TCS's operating margin in FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L09 | lookup | What was HCLTech's EBIT margin in FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L10 | lookup | What was Infosys's operating margin in FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L11 | lookup | How many employees did TCS have at the end of FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L12 | lookup | What was Infosys's headcount in FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L13 | lookup | What was HCLTech's attrition in FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| L14 | lookup | What was Infosys's voluntary attrition in FY2025-26? | answered ✓ | ✓ | – | 1/1 | facts | 0.0 |
| C01 | compare | Compare the revenue of TCS, Infosys and HCLTech in FY2025-26. | answered ✓ | ✓ | – | 3/3 | facts | 0.0 |
| C02 | compare | Compare the operating margins of the three companies. | answered ✓ | ✓ | – | 3/3 | facts | 0.0 |
| C03 | compare | Compare the headcount of TCS, Infosys and HCLTech. | answered ✓ | ✓ | – | 3/3 | facts | 0.0 |
| C04 | compare | Which company had the highest net profit in FY2025-26? | answered ✓ | ✓ | – | 3/3 | facts | 0.0 |
| D01 | descriptive | How does HCLTech describe its GenAI investments? | answered ✓ | – | ✓ | 2/2 | vllm | 4.5 |
| D02 | descriptive | What does TCS say about its AI strategy? | answered ✓ | – | ✓ | 0/0 | vllm | 5.8 |
| D03 | descriptive | How does Infosys describe its large deal wins in FY2025-26? | answered ✓ | – | ✓ | 2/2 | vllm | 4.1 |
| D04 | descriptive | What key risks does Infosys highlight? | answered ✓ | – | ✓ | 0/0 | vllm | 3.5 |
| D05 | descriptive | What dividend per share did Infosys declare for FY2025-26? | answered ✓ | ✗ | – | 3/3 | vllm | 4.7 |
| D06 | descriptive | How many US$50 million+ clients did Infosys have in FY2025-26? | answered ✓ | ✓ | – | 1/1 | vllm | 2.2 |
| D07 | descriptive | What was the large deal TCV of Infosys in FY2025-26? | answered ✓ | ✓ | – | 1/1 | vllm | 1.6 |
| U01 | unanswerable | What was TCS's revenue in FY2029-30? | not_found ✓ | – | – | 0/0 | vllm | 0.4 |
| U02 | unanswerable | What is the favourite cricket team of the TCS CEO? | not_found ✓ | – | – | 0/0 | vllm | 0.3 |
| U03 | unanswerable | What was Wipro's revenue in FY2025-26? | not_found ✓ | – | – | 0/0 | vllm | 0.0 |
| U04 | unanswerable | How many employees does HCLTech plan to hire in FY2030? | not_found ✓ | – | – | 0/0 | vllm | 0.0 |
| O01 | out_of_scope | Should I buy Infosys shares? | out_of_scope ✓ | – | – | 0/0 | vllm | 0.0 |
| O02 | out_of_scope | What will TCS's share price be next year? | out_of_scope ✓ | – | – | 0/0 | vllm | 0.0 |
