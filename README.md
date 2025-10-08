# SQLite — University Rankings (2012–2015)

This README documents how I connected to a pre-built SQLite database, ran basic analysis, and executed CRUD operations.

## Environment & Connection
- **Tool:** `sqlite3` CLI
- **Connect:** `sqlite3 university.db`
- nicer output inside the shell:
  ```sql
  .headers on
  .mode box
  .timer on
  ```

## View schema
```sql
CREATE TABLE university_rankings ( 
    world_rank        INTEGER, 
    institution       TEXT, 
    country           TEXT, 
    national_rank     INTEGER, 
    quality_of_education INTEGER, 
    alumni_employment    INTEGER, 
    quality_of_faculty   INTEGER, 
    publications         INTEGER, 
    influence            INTEGER, 
    citations            INTEGER, 
    broad_impact         INTEGER, 
    patents              INTEGER, 
    score                REAL, 
    year                 INTEGER 
);
```

## Basic Analysis (before modifications)
- **Total rows:** `2200`
- **Rows by year:**
  - 2012: `100`
  - 2013: `100`
  - 2014: `1000`
  - 2015: `1000`

- **Score stats by year:**
  - 2012 — min: `43.36`, max: `100.0`, avg: `54.941`
  - 2013 — min: `44.26`, max: `100.0`, avg: `55.271`
  - 2014 — min: `44.18`, max: `100.0`, avg: `47.271`
  - 2015 — min: `44.02`, max: `100.0`, avg: `46.864`

- **Global score stats:** min `43.36`, max `100.0`, sum `105156.47`, avg `47.798`

## CRUD Tasks

### 1) INSERT — Add new university (2014)
**Operation:** Insert *Duke Tech (USA)* with `world_rank=350`, `score=60.5`, `year=2014`.
```sql
BEGIN;
INSERT INTO university_rankings (institution, country, world_rank, score, year)
VALUES ('Duke Tech', 'USA', 350, 60.5, 2014);
SELECT changes() AS inserted_rows;   -- 1
COMMIT;
```
**Result:** 1 row inserted. Verified presence of the new record.

### 2) READ — Japan in global top 200 for 2013
**Question:** How many universities from Japan appear in the global top 200 in 2013?
```sql
SELECT COUNT(*) AS jp_top200_2013
FROM university_rankings
WHERE year = 2013 AND country = 'Japan' AND world_rank <= 200;
```
**Answer:** `6`

### 3) UPDATE — Correct Oxford 2014 score (+1.2)
**Operation:** Increase 2014 *University of Oxford* score by `+1.2`.
```sql
BEGIN;
UPDATE university_rankings
SET score = score + 1.2
WHERE institution = 'University of Oxford' AND year = 2014;
SELECT changes() AS updated_rows;    -- 1
COMMIT;
```
**Result:** 1 row updated.  
**Post-update check:**
```sql
SELECT institution, year, score
FROM university_rankings
WHERE institution='University of Oxford' AND year=2014;
```
**Observed:** score is now `98.71` → **previous score was `97.51`**.

### 4) DELETE — Remove 2015 rows with score < 45
**Policy:** Committee decided scores `< 45` in 2015 should be removed.
```sql
-- count before delete
SELECT COUNT(*) AS to_delete_2015_below_45
FROM university_rankings
WHERE year = 2015 AND score < 45;  -- 556

BEGIN;
DELETE FROM university_rankings
WHERE year = 2015 AND score < 45;
SELECT changes() AS deleted_rows;
COMMIT;
```
**Result:** `556` rows deleted from 2015.

### Net row count effect
- Start: `2200`
- After **INSERT**: `+1` → `2201`
- After **DELETE**: `-556` → **`1645` final rows**

Everything is reproducible from the `.sql` files included in this folder.
