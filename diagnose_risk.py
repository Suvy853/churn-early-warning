import sqlite3

conn = sqlite3.connect('data/churn_system.db')
cursor = conn.cursor()

# Check actual distribution of churn probabilities
cursor.execute('''
SELECT 
    SUM(CASE WHEN churn_probability < 0.20 THEN 1 ELSE 0 END) as under_20,
    SUM(CASE WHEN churn_probability >= 0.20 AND churn_probability < 0.40 THEN 1 ELSE 0 END) as between_20_40,
    SUM(CASE WHEN churn_probability >= 0.40 THEN 1 ELSE 0 END) as above_40
FROM predictions
''')

result = cursor.fetchone()
print(f"Under 20%: {result[0]}")
print(f"Between 20-40%: {result[1]}")
print(f"Above 40%: {result[2]}")

# Show some actual churn values > 0.40
cursor.execute('''
SELECT customer_id, churn_probability, risk_tier 
FROM predictions 
WHERE churn_probability > 0.40 
LIMIT 10
''')

print("\nCustomers with churn > 40%:")
rows = cursor.fetchall()
if rows:
    for row in rows:
        print(f"  {row[0]}: {row[1]:.4f} - {row[2]}")
else:
    print("  None found")

conn.close()