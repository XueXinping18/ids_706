-- 01_basic_analysis.sql
SELECT COUNT(*) AS total_rows FROM university_rankings;

SELECT year, COUNT(*) AS rows_in_year
FROM university_rankings
GROUP BY year
ORDER BY year;

SELECT MIN(score) AS min_score, MAX(score) AS max_score,
       SUM(score) AS sum_score, ROUND(AVG(score),3) AS avg_score
FROM university_rankings;

SELECT year,
       MIN(score) AS min_s, MAX(score) AS max_s,
       ROUND(AVG(score),3) AS avg_s
FROM university_rankings
GROUP BY year
ORDER BY year;
