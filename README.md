Perfect — you want a clean, submission-ready **Markdown report** that looks professional, satisfies all assignment requirements, and leaves space for screenshots after each query.
Here’s your fully rewritten version — polished, consistent in tone, and formatted for easy grading.
You can copy it directly into your submission file (`README_SQL_Guide.md`) or Notion.

---

# Personalized SQL Reference Guide (SQLite)

**Goal:**
This project practices intermediate to advanced SQL concepts, including table design, data manipulation, joins, aggregation, window functions, and common table expressions.
The guide documents both the SQL logic and outputs as a reusable reference for interviews and data analysis tasks.

---

## 1. Database Schema and Design Rationale

### Business Goal and Data Model

The database models a simplified **commerce system** that tracks customers, products, orders, and payments.
Key relationships:

* **Customers** place **Orders**.
* Each order contains multiple **Order Items** referencing **Products**.
* **Payments** record how orders are paid.
* **Employees** process orders and report to managers.

This design supports analytics such as customer spending, product sales, and employee hierarchy.

---

### Entities and Purpose

| Table           | Description                                                    |
| --------------- | -------------------------------------------------------------- |
| **customers**   | Stores buyer information such as name, email, city, and state. |
| **products**    | Product catalog containing item name, category, and price.     |
| **orders**      | Order header capturing date, status, and sales employee.       |
| **order_items** | Line-level order details: product, quantity, and unit price.   |
| **payments**    | Payment records linked to orders, allowing multiple methods.   |
| **employees**   | Employee data with `manager_id` for organizational hierarchy.  |

Each table keeps only attributes unique to that entity, minimizing redundancy.

---

### Normalization (≈ 3NF)

* Repeating groups separated into related tables (e.g., `order_items` for one-to-many).
* Attributes depend only on their primary key (no partial dependencies).
* Customer and product details are stored in their own tables to prevent duplication.
* Historical sale prices stored in `order_items.unit_price` ensure data accuracy over time.

---

### Keys and Relationships

* **Primary Keys:** `INTEGER` surrogates (auto-increment).
* **Foreign Keys:**

  * `orders.customer_id → customers.customer_id`
  * `orders.employee_id → employees.employee_id`
  * `order_items.order_id → orders.order_id`
  * `order_items.product_id → products.product_id`
  * `payments.order_id → orders.order_id`
  * `employees.manager_id → employees.employee_id`

**Cardinalities:**

* customers 1—* orders
* orders 1—* order_items
* products 1—* order_items
* orders 1—* payments
* employees 1—* employees (manager hierarchy)

---

### Constraints and Business Rules

* **NOT NULL** constraints for mandatory fields such as name and price.
* **CHECK** constraints to prevent invalid data (`price >= 0`, `quantity > 0`).
* **UNIQUE(email)** to avoid duplicate customer records.
* **ISO-8601 text dates** (e.g., `'2024-09-03'`) for easy date comparisons in SQLite.
* **RIGHT JOIN note:** SQLite does not support `RIGHT JOIN`.
  You can emulate it with a reversed `LEFT JOIN`:

  ```sql
  SELECT o.order_id, c.customer_id, c.first_name
  FROM orders o
  LEFT JOIN customers c ON c.customer_id = o.customer_id;
  ```

---

## 2. Practice Questions and Queries

Each query below answers a specific analytical or operational question about the dataset.
After running each SQL command, capture the result and paste its **screenshot** in the space provided.

---

### Q1. How can we update an order’s status once payment is confirmed?

This query updates orders that are still pending but already have a payment record, then verifies the result.

```sql
UPDATE orders
SET status = 'SHIPPED'
WHERE order_id = 104
  AND status = 'PENDING'
  AND EXISTS (SELECT 1 FROM payments p WHERE p.order_id = orders.order_id);

SELECT order_id, status FROM orders WHERE order_id = 104;
```

**Screenshot:**
*(Insert image `Q1.png` here)*

---

### Q2. Which are the top three most expensive products?

This query retrieves the highest-priced products in descending order.

```sql
SELECT product_id, name, category, price
FROM products
ORDER BY price DESC
LIMIT 3;
```

**Screenshot:**
*(Insert image `Q2.png` here)*

---

### Q3. Which customers have spent at least $300 in total payments?

This query aggregates total payments per customer and filters with `HAVING`.

```sql
SELECT c.customer_id,
       c.first_name || ' ' || c.last_name AS customer_name,
       ROUND(SUM(p.amount), 2) AS total_spent
FROM customers c
JOIN orders o   ON o.customer_id = c.customer_id
JOIN payments p ON p.order_id = o.order_id
GROUP BY c.customer_id
HAVING SUM(p.amount) >= 300
ORDER BY total_spent DESC;
```

**Screenshot:**
*(Insert image `Q3.png` here)*

---

### Q4. What products were included in each order, and what is each line’s total cost?

This query joins multiple tables to show detailed order lines and computed extended prices.

```sql
SELECT o.order_id, o.order_date,
       c.first_name || ' ' || c.last_name AS customer_name,
       pr.name AS product_name,
       oi.quantity, oi.unit_price,
       ROUND(oi.quantity * oi.unit_price, 2) AS extended_price
FROM orders o
JOIN customers c    ON c.customer_id = o.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products pr    ON pr.product_id = oi.product_id
ORDER BY o.order_id, oi.order_item_id;
```

**Screenshot:**
*(Insert image `Q4.png` here)*

---

### Q5. How can we show all customers, even those without any orders?

This query demonstrates a `LEFT JOIN` and explains how to emulate a `RIGHT JOIN` in SQLite.

```sql
SELECT c.customer_id,
       c.first_name || ' ' || c.last_name AS customer_name,
       o.order_id
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id
ORDER BY c.customer_id, o.order_id;
```

> **RIGHT JOIN equivalent (emulation):**
>
> ```sql
> SELECT o.order_id, c.customer_id, c.first_name
> FROM orders o
> LEFT JOIN customers c ON c.customer_id = o.customer_id;
> ```

**Screenshot:**
*(Insert image `Q5.png` here)*

---

### Q6. How can we make order statuses more readable and handle missing cities?

This query uses `CASE WHEN` and `COALESCE` to clean and standardize data.

```sql
SELECT o.order_id,
       CASE o.status
         WHEN 'SHIPPED'   THEN 'Completed'
         WHEN 'PENDING'   THEN 'Awaiting Payment/Action'
         WHEN 'CANCELLED' THEN 'Cancelled'
         WHEN 'REFUNDED'  THEN 'Refunded'
         ELSE 'Unknown'
       END AS status_label,
       COALESCE(c.city, 'Unknown City') AS city
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
ORDER BY o.order_id;
```

**Screenshot:**
*(Insert image `Q6.png` here)*

---

### Q7. Which customers spent the most, and how do they rank?

This query uses window functions to rank customers by spending and compare to previous totals.

```sql
WITH totals AS (
  SELECT c.customer_id,
         c.first_name || ' ' || c.last_name AS customer_name,
         ROUND(SUM(p.amount), 2) AS total_spent
  FROM customers c
  JOIN orders o   ON o.customer_id = c.customer_id
  JOIN payments p ON p.order_id = o.order_id
  GROUP BY c.customer_id
)
SELECT customer_id, customer_name, total_spent,
       RANK() OVER (ORDER BY total_spent DESC) AS spend_rank,
       LAG(total_spent) OVER (ORDER BY total_spent DESC) AS prev_total
FROM totals
ORDER BY spend_rank;
```

**Screenshot:**
*(Insert image `Q7.png` here)*

---

### Q8. What are the total amounts per order, and how is the employee hierarchy structured?

This includes both a standard and recursive CTE.

```sql
-- Order totals per order
WITH order_totals AS (
  SELECT o.order_id, SUM(oi.quantity * oi.unit_price) AS order_total
  FROM orders o
  JOIN order_items oi ON oi.order_id = o.order_id
  GROUP BY o.order_id
)
SELECT o.order_id, o.status, ROUND(ot.order_total, 2) AS order_total
FROM orders o
JOIN order_totals ot ON ot.order_id = o.order_id
ORDER BY o.order_id;

-- Recursive CTE for hierarchy
WITH RECURSIVE hierarchy AS (
  SELECT employee_id, full_name, role, manager_id, 0 AS level
  FROM employees
  WHERE manager_id IS NULL
  UNION ALL
  SELECT e.employee_id, e.full_name, e.role, e.manager_id, h.level + 1
  FROM employees e
  JOIN hierarchy h ON e.manager_id = h.employee_id
)
SELECT * FROM hierarchy ORDER BY level, employee_id;
```

**Screenshot:**
*(Insert image `Q8.png` here)*

---

### Q9. How can we extract email domains and group orders by month?

This query applies string and date functions for data transformation.

```sql
-- Extract email domain
SELECT c.customer_id, c.first_name, c.last_name,
       lower(substr(c.email, instr(c.email, '@') + 1)) AS email_domain
FROM customers c
ORDER BY c.customer_id;

-- Orders grouped by month
SELECT strftime('%Y-%m', o.order_date) AS order_month,
       COUNT(*) AS orders_count
FROM orders o
GROUP BY strftime('%Y-%m', o.order_date)
ORDER BY order_month;
```

**Screenshot:**
*(Insert image `Q9.png` here)*

---

### Q10. Which cities appear in both sets, and who has not placed any orders?

This query uses `UNION` and `EXCEPT` to combine and exclude datasets.

```sql
-- Combine distinct city sets
SELECT DISTINCT city FROM customers WHERE state = 'NC'
UNION
SELECT 'Charlotte' AS city;

-- Customers without orders
SELECT customer_id FROM customers
EXCEPT
SELECT DISTINCT customer_id FROM orders
ORDER BY customer_id;
```

**Screenshot:**
*(Insert image `Q10.png` here)*


