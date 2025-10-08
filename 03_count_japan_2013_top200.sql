-- 03_count_japan_2013_top200.sql
SELECT COUNT(*) AS jp_top200_2013
FROM university_rankings
WHERE year = 2013 AND country = 'Japan' AND world_rank <= 200;
