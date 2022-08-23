-- 1757. Recyclable and Low Fat Products
# Write your MySQL query statement below
Select product_id from Products
where (low_fats = 'Y') and (recyclable = 'Y')

-- 1741. Find Total Time Spent by Each Employee
# Write your MySQL query statement below
SELECT event_day as day, emp_id, sum(out_time - in_time) as total_time
FROM Employees
GROUP BY emp_id, day

-- 1683. Invalid Tweets
# Write your MySQL query statement below
SELECT  tweet_id
FROM Tweets
WHERE length(content) > 15

-- 1303. Find the Team Size
# Write your MySQL query statement below
SELECT employee_id, (SELECT count(team_id)  FROM Employee e1
                     WHERE e1.team_id = e2.team_id) as team_size
FROM Employee e2

-- 2356. Number of Unique Subjects Taught by Each Teacher
# Write your MySQL query statement below
SELECT teacher_id, count(distinct(subject_id)) as cnt 
from Teacher
GROUP by teacher_id

-- 2339. All the Matches of the League
# Write your MySQL query statement below
SELECT a.team_name as home_team, b.team_name as away_team 
        from Teams a, Teams b
        WHERE a.team_name != b.team_name
-- 1378. Replace Employee ID With The Unique Identifier
# Write your MySQL query statement below
SELECT IFNULL(unique_id,null) as unique_id, name
FROM Employees
LEFT JOIN EmployeeUNI
ON Employees.id = EmployeeUNI.id

-- 1693. Daily Leads and Partners
# Write your MySQL query statement below
SELECT date_id, make_name, count(distinct(lead_id)) as unique_leads, count(distinct(partner_id)) as unique_partners
FROM DailySales
Group By make_name,date_id
