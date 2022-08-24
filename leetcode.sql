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
