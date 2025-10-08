-- 04_update_oxford_2014_score.sql
BEGIN;
UPDATE university_rankings
SET score = score + 1.2
WHERE institution = 'University of Oxford' AND year = 2014;
SELECT changes() AS updated_rows;
COMMIT;

-- verification
SELECT institution, year, score
FROM university_rankings
WHERE institution='University of Oxford' AND year=2014;
