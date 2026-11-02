# Decision log
One line per decision: date | decision | reason | rubric criterion it supports.

- 2026-11-02 | Gate 0 passed: 1,017,209 rows, 1,115 stores, 2013-01-01 to 2015-07-31, no duplicate store-days | matches the competition description | Data understanding
- 2026-11-02 | 180 stores have no rows from 2014-07-01 to 2014-12-31 (184 days), plus one missing row in Jan 2013 | consistent with the refurbishment closures in the competition notes | Data understanding
- 2026-11-02 | Missing store-days are treated as missing, not zero; lag and rolling features use only days that exist, and are left empty if a window has fewer than 10 days | zeros would fake a collapse in sales | Preprocessing
- 2026-11-02 | A row is scored only if the seasonal naive forecast exists for it; every model is scored on the same rows and the number of excluded rows is reported | fair comparison on identical rows | Evaluation design
- 2026-11-02 | Closed days (17.0 percent of rows, all with zero sales) are excluded from training and scoring and set to zero by the opening schedule | a schedule fact, not a forecasting problem | Problem formulation
- 2026-11-02 | 54 open days with zero sales are dropped from training and scoring | unexplained and too few to matter | Preprocessing
- 2026-11-02 | StateHoliday is read as text with codes 0 none, a public holiday, b Easter, c Christmas | the raw column mixes the number 0 and the text 0 | Preprocessing
- 2026-11-02 | CompetitionDistance (3 stores) filled with the training median plus a missing flag | few missing values | Preprocessing
- 2026-11-02 | CompetitionOpenSince (354 stores) left empty with a flag, not filled with zero | the opening date is unknown | Preprocessing
- 2026-11-02 | Promo2 fields (544 stores) marked not applicable and Promo2 active set to 0 | empty because the store is not in Promo2, not an error | Preprocessing
- 2026-11-02 | Target is log1p(Sales); errors reported back in euros | skew 1.59 raw against -0.11 after log1p on open days | Problem formulation
- 2026-11-02 | Customers is excluded from features | only known after the day ends; correlation with Sales is 0.82 on open days, and a leakage demo will measure the inflation | Preprocessing and leakage
- 2026-11-02 | The refurbishment gap overlaps folds F1 to F4; those stores have no validation rows then and limited history in early 2015; reported as a failure-audit slice | known data limitation | Evaluation design
- 2026-11-02 | Warehouse is a star schema in SQLite, loaded from staging tables; the first 30 days load one at a time, the rest in one bulk batch | proves the daily process and keeps the raw file untouched | Reproducibility
- 2026-11-02 | Seven reconciliation checks run after every batch and are stored in recon_results; a failure stops the run | catches loading errors early | Reproducibility
- 2026-11-02 | Closed days stay in the fact table with open_flag 0; they are excluded later from training and scoring | the warehouse keeps every real event | Problem formulation
- 2026-11-02 | Missing store attributes stay empty in the warehouse; imputation happens later using training rows only | avoids learning from test data | Preprocessing