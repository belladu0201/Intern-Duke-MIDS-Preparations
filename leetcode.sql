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
