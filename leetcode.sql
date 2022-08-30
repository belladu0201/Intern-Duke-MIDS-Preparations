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
-- 1795. Rearrange Products Table
# Write your MySQL query statement below
SELECT * FROM (
    SELECT product_id, 'store1' as store, store1 as price from products 
    union
    SELECT product_id, 'store2' as store, store2 as price from products
    union
    SELECT product_id, 'store3' as store, store3 as price from products
) cur
where cur.price is not null -- != null is different from is not null
-- 1350. Students With Invalid Departments
# Write your MySQL query statement below
SELECT Students.id, Students.name 
FROM Students
LEFT JOIN Departments
on Students.department_id = Departments.id
where Departments.id is null;

-- 1587. Bank Account Summary II
# Write your MySQL query statement below
SELECT name, sum(amount) as balance from Users
Inner Join Transactions
On Users.account = Transactions.account
Group by Transactions.account
having balance > 10000

-- 1571. Warehouse Manager
# Write your MySQL query statement below
SELECT name as warehouse_name, sum(Warehouse.units*(Width*Length*Height)) as volume
FROM Warehouse
INNER JOIN Products
ON Warehouse.product_id = Products.product_id
GROUP BY Warehouse.name

-- 182. Duplicate Emails
# Write your MySQL query statement below
SELECT email as Email from Person
Group By email
having count(Email) > 1
-- 183. Customers Who Never Order
# Write your MySQL query statement below
SELECT name as Customers
FROM Customers
WHERE Customers.id NOT IN (SELECT customerId FROM Orders)
-- 577. Employee Bonus
# Write your MySQL query statement below
SELECT name, bonus 
FROM Employee
LEFT JOIN Bonus
ON Employee.empId = Bonus.empId
WHERE bonus < 1000 or bonus is Null
-- 196. Delete Duplicate Emails
# Please write a DELETE statement and DO NOT write a SELECT statement.
# Write your MySQL query statement below
DELETE a from Person a
Inner Join Person b
ON a.email = b.email
WHERE a.id > b.id
-- ###########
# Please write a DELETE statement and DO NOT write a SELECT statement.
# Write your MySQL query statement below
DELETE a
FROM Person a, Person b
-- compare two table, make sure their email address same, we need to delete the one with higher id
where a.email = b.email and a.id > b.id

-- 197. Rising Temperature
# Write your MySQL query statement below
SELECT a.id
FROM Weather a, Weather b
WHERE datediff(a.recordDate, b.recordDate) = 1 and a.temperature > b.temperature

-- 511. Game Play Analysis I
# Write your MySQL query statement below
SELECT DISTINCT player_id, min(event_date) as first_login
FROM Activity
GROUP BY player_id
-- ORDER BY (event_date) ASC

-- 512. Game Play Analysis II
# Write your MySQL query statement below
SELECT player_id, device_id
FROM Activity
WHERE (event_date, player_id) in (SELECT  
                                        MIN(event_date), 
                                        player_id 
                                  FROM activity 
                                  GROUP BY player_id)
-- GROUP BY player_id
-- ORDER BY event_date ASC
-- having min(event_date)

-- 584. Find Customer Referee
# Write your MySQL query statement below
SELECT name
FROM Customer
where referee_id != 2 or referee_id is null


-- 586. Customer Placing the Largest Number of Orders
# Write your MySQL query statement below
-- SELECT max(customer_number) as customer_number
-- FROM Orders
-- LIMIT 1;

--# SELECT customer_number
--# FROM Orders
--# ORDER BY customer_number DESC
--# LIMIT 1;
SELECT customer_number
FROM (SELECT customer_number,count(order_number) as c 
FROM Orders
GROUP BY customer_number)x
ORDER BY c DESC
LIMIT 1;

-- 596. Classes More Than 5 Students
# Write your MySQL query statement below
SELECT class FROM (SELECT class, COUNT(student) FROM Courses
                  GROUP BY class having COUNT(student) >= 5) x


-- 603. Consecutive Available Seats
# Write your MySQL query statement below
SELECT DISTINCT c1.seat_id FROM Cinema c1, Cinema c2
WHERE(abs(c1.seat_id - c2.seat_id) = 1) AND c1.free = 1 and c2.free = 1
ORDER BY seat_id ASC


-- 613. Shortest Distance in a Line
# Write your MySQL query statement below
SELECT MIN(abs(p1.x - p2.x)) as shortest from Point p1, Point p2
WHERE p1.x != p2.x613. Shortest Distance in a Line

-- 607. Sales Person
# Write your MySQL query statement below
SELECT name 
FROM SalesPerson
WHERE sales_id NOT IN (
    SELECT sales_id FROM Orders
    INNER JOIN Company
    ON Company.com_id = Orders.com_id and Company.name = 'RED'
)

-- 627. Swap Salary
# Write your MySQL query statement below
UPDATE salary SET sex = if(sex = 'm','f', 'm')

-- 2026. Low-Quality Problems
# Write your MySQL query statement below
SELECT problem_id FROM Problems
WHERE (likes/(likes + dislikes)) * 100 < 60
ORDER BY problem_id ASC

-- 1821. Find Customers With Positive Revenue this Year
# Write your MySQL query statement below
SELECT customer_id
FROM Customers
WHERE year = 2021 and revenue > 0

-- 1484. Group Sold Products By The Date
# Write your MySQL query statement below
select sell_date, count(distinct product) as num_sold,
group_concat(distinct product order by product) as products
FROM Activities
GROUP BY sell_date

-- 2377. Sort the Olympic Table
# Write your MySQL query statement below
SELECT *
FROM Olympic
ORDER BY gold_medals desc,silver_medals desc,bronze_medals desc, country asc
