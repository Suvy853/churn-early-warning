import sqlite3

conn = sqlite3.connect('data/churn_system.db')
cursor = conn.cursor()

# Check churn probability distribution
cursor.execute('''
SELECT 
    COUNT(*) as total,
    MIN(churn_probability) as min_churn,
    MAX(churn_probability) as max_churn,
    AVG(churn_probability) as avg_churn
FROM predictions
''')

result = cursor.fetchone()
print(f"Total predictions: {result[0]}")
print(f"Min churn probability: {result[1]:.4f}")
print(f"Max churn probability: {result[2]:.4f}")
print(f"Avg churn probability: {result[3]:.4f}")

# Check risk tier distribution
cursor.execute('''
SELECT risk_tier, COUNT(*) FROM predictions GROUP BY risk_tier
''')
print("\nRisk Tier Distribution:")
for tier, count in cursor.fetchall():
    print(f"  {tier}: {count}")

conn.close()