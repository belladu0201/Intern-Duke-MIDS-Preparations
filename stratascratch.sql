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
