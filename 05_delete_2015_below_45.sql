-- 05_delete_2015_below_45.sql
-- pre-check (optional)
SELECT COUNT(*) AS to_delete_2015_below_45
FROM university_rankings
WHERE year = 2015 AND score < 45;

BEGIN;
DELETE FROM university_rankings
WHERE year = 2015 AND score < 45;
SELECT changes() AS deleted_rows;
COMMIT;
