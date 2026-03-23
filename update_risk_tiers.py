import sqlite3

conn = sqlite3.connect('data/churn_system.db')
cursor = conn.cursor()

# Update risk tier based on new thresholds
cursor.execute('''
UPDATE predictions SET risk_tier = CASE 
    WHEN churn_probability < 0.20 THEN 'Low Risk'
    WHEN churn_probability < 0.40 THEN 'Medium Risk'
    ELSE 'High Risk'
END
''')

conn.commit()
print(f"✓ Updated {cursor.rowcount} predictions with new risk tiers")

# Verify
cursor.execute('''
SELECT risk_tier, COUNT(*) FROM predictions GROUP BY risk_tier
''')

print("\nUpdated Risk Tier Distribution:")
for tier, count in cursor.fetchall():
    print(f"  {tier}: {count}")

conn.close()