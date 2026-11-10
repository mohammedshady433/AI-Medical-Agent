# Try the library in five minutes

These are invented examples, safe to use for a demonstration. Start the core with `scripts/test_console.py`.

1. In the test console, open **Documents** and upload `products.csv`, `staff-policy.txt` and `questions.jsonl`.
2. Wait for `ready` (`keyword_only` still works for exact words).
3. Optional file instruction: `PATCH /documents/<id> {"instructions": "Use this file for product price and warranty questions. Prices are in EGP. Do not invent missing products.", "enabled": true}`.
4. Optional general instruction: `PATCH /knowledge {"instructions": "Use my library for company policy and product questions. Cite the source and tell me when information is missing."}`.
5. Start a **New chat**. Ask "What is the price and warranty of Cedar Desk?" Expected: 7250 EGP and 24 months, with a source.
6. Ask "What is the annual training allowance?" Expected: 8000 EGP; approval before booking.
7. Ask "What is our parental leave entitlement?" Expected: the files don't give it.
8. Restart the core. Your files and instructions should still be there.

CSV/TSV needs unique column names in row one and consistent row lengths. TXT and JSONL should use UTF-8. JSONL means one complete JSON value per line. XLSX uses saved cell values, not recalculated formulas. Scans need OCR. Search is not a full-table accounting engine.
