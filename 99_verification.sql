-- 99_verification.sql
-- Final checks
SELECT world_rank, institution, country, score, year
FROM university_rankings
WHERE institution='Duke Tech' AND year=2014;

SELECT institution, year, score
FROM university_rankings
WHERE institution='University of Oxford' AND year=2014;

SELECT COUNT(*) AS remaining_below_45_2015
FROM university_rankings
WHERE year = 2015 AND score < 45;

SELECT COUNT(*) AS final_total_rows
FROM university_rankings;
