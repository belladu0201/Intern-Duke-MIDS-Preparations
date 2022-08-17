-- Churro Activity Date
select activity_date,  pe_description
from los_angeles_restaurant_health_inspections
WHERE facility_name = 'STREET CHURROS'
    and score < 95
   
   
--Highest Salary
SELECT 
    first_name, salary
from 
    worker
WHERE salary = (SELECT max(salary) from worker)

--Find the number of workers by department (Amazon)
select  
    department, COUNT(worker_id) as worker_count
from 
    worker
GROUP BY 
    department
ORDER BY
    worker_count DESC
-- Salaries Differences
-- Interview Question Date: November 2020
select (
    MAX(Case when department = 'marketing' then salary end) 
    - 
    MAX( Case when department = 'engineering' then salary end)) as s_diff
FROM db_employee de 
LEFT JOIN db_dept dd ON de.department_id = dd.id;


-- Departments With 5 Employees
-- SELECT department
-- from employee
-- group by department
-- having count(*) > 5
SELECT
    department
FROM
    employee
GROUP BY department
HAVING COUNT(DISTINCT id) >= 5
