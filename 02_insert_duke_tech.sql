-- 02_insert_duke_tech.sql
BEGIN;
INSERT INTO university_rankings (institution, country, world_rank, score, year)
VALUES ('Duke Tech', 'USA', 350, 60.5, 2014);
SELECT changes() AS inserted_rows;
COMMIT;

-- verification
SELECT world_rank, institution, country, score, year
FROM university_rankings
WHERE institution='Duke Tech' AND year=2014;
